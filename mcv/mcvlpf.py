#  MultiColor Exoplanet Validation tool
#  Copyright (C) 2021 Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable
from collections import namedtuple

import pandas as pd
import etta

from ldtk import LDPSetCreator
from matplotlib.pyplot import subplots, setp
from numpy import arange, zeros, concatenate, sqrt, ndarray, inf, atleast_2d, sum, median, where, floor, squeeze, argsort, isnan
from pytransit import sdss_g, sdss_r, sdss_i, sdss_z, RoadRunnerModel
from pytransit.contamination import Instrument, SMContamination
from pytransit.lpf.cntlpf import contaminate, PhysContLPF
from pytransit.lpf.loglikelihood import WNLogLikelihood, CeleriteLogLikelihood
from pytransit.lpf.tess.tgclpf import BaseTGCLPF
from pytransit.lpf.tesslpf import downsample_time
from pytransit.orbits import as_from_rhop, i_from_ba
from pytransit.utils.misc import fold
from uncertainties import ufloat

from .io.muscat2 import read_m2_data
from .io.photometry import Photometry
from .io.tess import read_tess
from .plotting import _jplot

TOI = namedtuple('TOI', 'tic toi tmag ra dec epoch period duration depth'.split())

def get_toi(toi):
    dtoi = etta.download_toi(toi=toi)
    tic, tmag, ra, dec = dtoi['TIC ID, TESS Mag, RA, Dec'.split(', ')].values[0]
    zero_epoch = ufloat(*dtoi[['Epoch (BJD)', 'Epoch (BJD) err']].values[0])
    period = ufloat(*dtoi[['Period (days)', 'Period (days) err']].values[0])
    duration = ufloat(*dtoi[['Duration (hours)', 'Duration (hours) err']].values[0])
    depth = ufloat(*dtoi[['Depth (ppm)', 'Depth (ppm) err']].values[0])
    return TOI(tic, toi, tmag, ra, dec, epoch=zero_epoch,
               period=period, duration=duration, depth=depth)


@dataclass
class Star:
    radius: ufloat
    teff: ufloat
    logg: ufloat
    z: ufloat

    @staticmethod
    def from_toi(toi: TOI):
        from astroquery.mast import Catalogs
        tb = Catalogs.query_object(f"TIC {toi.tic}", radius=.002, catalog="TIC")[0]
        radius = ufloat(*tb['rad e_rad'.split()])
        teff = ufloat(*tb['Teff e_Teff'.split()])
        logg = ufloat(*tb['logg e_logg'.split()])

        z = ufloat(*tb['MH e_MH'.split()])
        if isnan(z.n):
            z = ufloat(0, 0.01)
        return Star(radius, teff, logg, z)


class MCVLPF(BaseTGCLPF):
    def __init__(self, toi: float, star: Optional[Star] = None, split_transits: bool = False,
                 zero_epoch: Optional[ufloat] = None, period: Optional[ufloat] = None,
                 use_gp: bool = True, use_ldtk: bool = True, use_opencl: bool = False, use_pdc: bool = True,
                 heavy_baseline: bool = False, downsample: Optional[float] = None,
                 m2_passbands: Iterable = ('g', 'r', 'i', 'z_s')):

        name = f"TOI-{toi}"
        self.toi: TOI = get_toi(toi)
        self.zero_epoch = zero_epoch or self.toi.epoch
        self.period = period or self.toi.period
        self.star = star or Star.from_toi(self.toi)
        self.split_transits = split_transits

        self.use_gp = use_gp
        self.use_pdc = use_pdc
        self.use_opencl = use_opencl
        self.heavy_baseline = heavy_baseline
        self.downsample = downsample
        self.m2_passbands = m2_passbands
        tm = RoadRunnerModel('power-2-pm', small_planet_limit=0.005, parallel=True)

        self.result_dir = Path('.')
        self._stess = None
        self._ntess = None
        self.data: Optional[Photometry] = None

        times, fluxes, pbnames, pbs, wns, covs = self.read_data()
        pbids = pd.Categorical(pbs, categories=pbnames).codes
        wnids = arange(len(times))
        tref = floor(concatenate(times).min())

        self.wns = wns
        PhysContLPF.__init__(self, name, passbands=pbnames, times=times, fluxes=fluxes, pbids=pbids, wnids=wnids,
                             covariates=covs, tref=tref, tm=tm, nsamples=self.data.nsamples, exptimes=self.data.exptime)

        if use_ldtk:
            self.set_ldtk_priors()


    def read_data(self):
        ddata = Path('photometry')
        dtess = ddata/'tess'
        dm2 = ddata/'m2'
        m2_files = sorted(dm2.glob('*.fits'))
        dtess = read_tess(self.toi.tic, dtess, split_transits=self.split_transits, use_pdc=self.use_pdc,
                          zero_epoch=self.zero_epoch.n, period=self.period.n, depth=self.toi.depth.n*1e-6,
                          transit_duration=self.toi.duration.n/24, baseline_duration=4*self.toi.duration.n/24)
        dm2 = read_m2_data(m2_files, downsample=self.downsample, passbands=self.m2_passbands, heavy_baseline=self.heavy_baseline)
        pbnames = 'tess g r i z_s'.split()
        self._stess = len(dtess.time)
        self._ntess = sum([t.size for t in dtess.time])
        self.data = data = dtess + dm2
        data.fluxes = [f / median(f) for f in data.flux]
        data.covariates = [(c-c.mean(0)) / c.std(0) for c in data.covariates]
        return data.time, data.flux, pbnames, data.passband, data.noise, data.covariates

    def _init_instrument(self):
        """Set up the instrument and contamination model."""
        self.instrument = Instrument('example', [sdss_g, sdss_r, sdss_i, sdss_z])
        self.cm = SMContamination(self.instrument, "i'")
        self.add_prior(lambda pv: where(pv[:, 4] < pv[:, 5], 0, -inf))
        self.add_prior(lambda pv: where(pv[:, 8] < pv[:, 5], 0, -inf))

    def _init_lnlikelihood(self):
        if self.use_gp:
            self._add_lnlikelihood_model(CeleriteLogLikelihood(self, noise_ids=self.noise_ids[:self._stess]))
            self._add_lnlikelihood_model(WNLogLikelihood(self, noise_ids=self.noise_ids[self._stess:]))
        else:
            self._add_lnlikelihood_model(WNLogLikelihood(self))

    def _post_initialisation(self):
        if self.use_opencl:
            self.tm = self.tm.to_opencl()
        self.set_prior('tc', 'NP', self.zero_epoch.n, 2*self.zero_epoch.s)
        self.set_prior('p', 'NP', self.period.n, 2*self.period.s)
        self.set_prior('rho', 'UP', 1, 35)
        self.set_prior('k2_app', 'UP', round(0.5*self.toi.depth.n*1e-6, 5), round(1.5*self.toi.depth.n*1e-6, 5))
        self.set_prior('k2_true', 'UP', 0.02 ** 2, 0.95 ** 2)
        self.set_prior('k2_app_tess', 'UP', round(0.5*self.toi.depth.n*1e-6, 5), round(1.5*self.toi.depth.n*1e-6, 5))
        self.set_prior('teff_h', 'NP', self.star.teff.n, self.star.teff.s)
        self.set_prior('teff_c', 'UP', 2500, 12000)

    def set_ldtk_priors(self):
        from ldtk import tess, sdss_g, sdss_r, sdss_i, sdss_z
        star = self.star
        filters = [tess, sdss_g, sdss_r, sdss_i, sdss_z]
        sc = LDPSetCreator((star.teff.n, star.teff.s), (star.logg.n, star.logg.s), (star.z.n, star.z.s), filters)
        ps = sc.create_profiles(500)
        ps.set_uncertainty_multiplier(5)
        ldc, lde = ps.coeffs_p2mp()
        for i, p in enumerate(self.ps[self._sl_ld]):
            self.set_prior(p.name, 'NP', round(ldc.flat[i], 5), round(lde.flat[i], 5))

    def create_pv_population(self, npv: int = 50) -> ndarray:
        pvp = super().create_pv_population(npv)
        for p in self.ps[self._sl_lm]:
            if 'lm_i' in p.name:
                pvp[:, p.pid] = 0.01 * (pvp[:, p.pid] - 1.0) + 1.0
            else:
                pvp[:, p.pid] *= 0.01
        return pvp

    def transit_model(self, pvp):
        pvp = atleast_2d(pvp)
        cnt = zeros((pvp.shape[0], self.npb))
        zero_epoch = pvp[:,0] - self._tref
        period = pvp[:,1]
        smaxis = as_from_rhop(pvp[:, 2], period)
        inclination  = i_from_ba(pvp[:, 3], smaxis)
        radius_ratio = sqrt(pvp[:,5:6])
        ldc = pvp[:, self._sl_ld].reshape([-1, self.npb, 2])
        flux = self.tm.evaluate(radius_ratio, ldc, zero_epoch, period, smaxis, inclination)
        cnt[:, 0] = 1 - pvp[:, 8] / pvp[:, 5]
        cnref = 1. - pvp[:, 4] / pvp[:, 5]
        cnt[:, 1:] = self.cm.contamination(cnref, pvp[:, 6], pvp[:, 7])
        return contaminate(flux, cnt, self.lcids, self.pbids)

    def plot_folded_tess_transit(self, solution: str = 'de', pv: ndarray = None, binwidth: float = 1,
                                 plot_model: bool = True, plot_unbinned: bool = True, plot_binned: bool = True,
                                 xlim: tuple = None, ylim: tuple = None, ax=None, figsize: tuple = None):

        if pv is None:
            if solution.lower() == 'local':
                pv = self._local_minimization.x
            elif solution.lower() in ('de', 'global'):
                pv = self.de.minimum_location
            elif solution.lower() in ('mcmc', 'mc'):
                pv = self.posterior_samples().median().values
            else:
                raise NotImplementedError("'solution' should be either 'local', 'global', or 'mcmc'")

        if ax is None:
            fig, ax = subplots(figsize=figsize)
        else:
            fig, ax = None, ax

        ax.autoscale(enable=True, axis='x', tight=True)

        etess = self._ntess
        t = self.timea[:etess]
        fo = self.ofluxa[:etess]
        fm = squeeze(self.transit_model(pv))[:etess]
        bl = squeeze(self.baseline(pv))[:etess]

        phase = 24 * pv[1] * (fold(t, pv[1], pv[0], 0.5) - 0.5)
        sids = argsort(phase)
        phase = phase[sids]
        bp, bf, be = downsample_time(phase, (fo / bl)[sids], binwidth / 60)
        if plot_unbinned:
            ax.plot(phase, (fo / bl)[sids], 'k.', alpha=1, ms=2)
        if plot_binned:
            ax.errorbar(bp, bf, be, fmt='ko', ms=3)
        if plot_model:
            ax.plot(phase, fm[sids], 'k')
        setp(ax, ylim=ylim, xlim=xlim, xlabel='Time - T$_c$ [h]', ylabel='Normalised flux')

        if fig is not None:
            fig.tight_layout()
        return fig


    def plot_joint_marginals(self, figsize=None, nb=30, gs=25, with_contamination=False, **kwargs):
        df = self.posterior_samples()
        if with_contamination:
            xlabels = ['$\Delta$ T$_\mathrm{Eff}$ [K]', 'Apparent radius ratio', 'Ref. pb. contamination',
                   'TESS contamination', 'Impact parameter', 'Stellar density [g/cm$^3$]']
            return _jplot([df.teff_c - df.teff_h, df.k_app, df.cref, df.ctess, df.b, df.rho], df.k_true,
                          xlabels, 'True radius ratio', figsize, nb, gs, **kwargs)
        else:
            xlabels = ['$\Delta$ T$_\mathrm{Eff}$ [K]', 'Apparent radius ratio',
                       'Impact parameter', 'Stellar density [g/cm$^3$]']
            return _jplot([df.teff_c - df.teff_h, df.k_app, df.b, df.rho], df.k_true,
                          xlabels, 'True radius ratio', figsize, nb, gs, **kwargs)