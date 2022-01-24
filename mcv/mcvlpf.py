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
from numpy import arange, zeros, concatenate, sqrt, ndarray, inf, atleast_2d, sum, median, where, floor, squeeze, \
    argsort, isnan, repeat, array, unique, ceil, percentile, poly1d, newaxis
from numpy.random import uniform, permutation
from pytransit import sdss_g, sdss_r, sdss_i, sdss_z, RoadRunnerModel
from pytransit.contamination import Instrument, SMContamination
from pytransit.lpf.cntlpf import contaminate, PhysContLPF
from pytransit.lpf.loglikelihood import WNLogLikelihood, CeleriteLogLikelihood
from pytransit.lpf.tess.tgclpf import BaseTGCLPF
from pytransit.lpf.tesslpf import downsample_time
from pytransit.orbits import as_from_rhop, i_from_ba, i_from_baew, d_from_pkaiews, epoch
from pytransit.param import PParameter, GParameter, UniformPrior as UP, NormalPrior as NP
from pytransit.utils.misc import fold
from uncertainties import ufloat

from .io import Photometry, read_tess, read_m2, read_lco, read_m3
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
    def __init__(self, toi: float, star: Optional[Star] = None,
                 zero_epoch: Optional[ufloat] = None, period: Optional[ufloat] = None, model_ttvs: bool = False,
                 absolute_contamination: bool = False,
                 use_gp: bool = True, use_ldtk: bool = True, use_opencl: bool = False):

        self.name = f"TOI-{toi}"
        self.toi: TOI = get_toi(toi)
        self.zero_epoch = zero_epoch or self.toi.epoch
        self.period = period or self.toi.period
        self.star = star or Star.from_toi(self.toi)

        self.absolute_contamination: bool = absolute_contamination
        self.model_ttvs: bool = model_ttvs
        self.use_gp: bool = use_gp
        self.use_opencl: bool = use_opencl
        self.use_ldtk: bool = use_ldtk

        self.data: Optional[Photometry] = Photometry([], [], [], [], [], [], [], [], [], [])
        self._stess = None
        self._ntess = None
        self._setup_finished: bool = False

    def finish_setup(self):
        data = self.data
        data.flux = [f / median(f) for f in data.flux]
        data.covariates = [(c-c.mean(0)) / c.std(0) for c in data.covariates]
        pbnames = 'tess g r i z_s'.split()

        pbids = pd.Categorical(data.passband, categories=pbnames).codes
        wnids = arange(len(data.time))
        tref = floor(concatenate(data.time).min())

        if self.model_ttvs:
            epochs = array([epoch(t, self.zero_epoch.n, self.period.n)[0] for t in data.time])
            unique_epochs = [epochs[0]]
            epids = [0]
            epid = 0
            ec = epochs[0]
            for ep in epochs[1:]:
                if ec != ep:
                    unique_epochs.append(ep)
                    epid += 1
                    ec = ep
                epids.append(epid)
            self.nep = epid + 1
            self.epochs = array(unique_epochs)
        else:
            self.epochs = zeros(1, int)
            self.nep = 1
            epids = zeros(len(data.time), int)

        self.wns = data.noise
        tm = RoadRunnerModel('power-2-pm', small_planet_limit=0.005, parallel=True)
        PhysContLPF.__init__(self, self.name, passbands=pbnames, times=data.time, fluxes=data.flux, pbids=pbids, wnids=wnids,
                             covariates=data.covariates, tref=tref, tm=tm, nsamples=self.data.nsamples, exptimes=self.data.exptime,
                             result_dir=Path('.'))

        if self.model_ttvs:
            self.tm.epids = array(epids)

        if self.use_ldtk:
            self.set_ldtk_priors()

    def read_tess(self, datadir: Optional[Path] = None, split_transits: bool = False, use_pdc: bool = True):
        if not datadir:
            datadir = Path('photometry/tess')
        dtess = read_tess(self.toi.tic, datadir, split_transits=split_transits, use_pdc=use_pdc,
                          zero_epoch=self.zero_epoch.n, period=self.period.n, depth=self.toi.depth.n*1e-6,
                          transit_duration=self.toi.duration.n/24, baseline_duration=4*self.toi.duration.n/24)
        self._stess = len(dtess.time)
        self._ntess = sum([t.size for t in dtess.time])
        self.data += dtess

    def read_m2(self, datadir: Optional[Path] = None, heavy_baseline: bool = False,
                downsample: Optional[float] = None, passbands: Iterable = ('g', 'r', 'i', 'z_s')):
        if not datadir:
            datadir = Path('photometry/m2')
        files = sorted(datadir.glob('*.fits'))
        self.data += read_m2(files, downsample=downsample, passbands=passbands, heavy_baseline=heavy_baseline)

    def read_m3(self, datadir: Optional[Path] = None, heavy_baseline: bool = False):
        if not datadir:
            datadir = Path('photometry/lco')
        self.data += read_m3(datadir, heavy_baseline=heavy_baseline)

    def read_lco_file(self, fname: Path, passband: str, instrument: str):
        self.data += read_lco(fname, passband, instrument)

    def _init_instrument(self):
        """Set up the instrument and contamination model."""
        self.instrument = Instrument('example', [sdss_g, sdss_r, sdss_i, sdss_z])
        self.cm = SMContamination(self.instrument, "i'")

    def _init_p_orbit(self):
        """Orbit parameter initialisation.
        """
        porbit = [
            GParameter('p', 'period', 'd', NP(1.0, 1e-5), (0, inf)),
            GParameter('rho', 'stellar_density', 'g/cm^3', UP(0.1, 25.0), (0, inf)),
            GParameter('b', 'impact_parameter', 'R_s', UP(0.0, 1.0), (0, 1))]
        self.ps.add_global_block('orbit', porbit)

        if self.model_ttvs:
            ptc = [GParameter(f'tc_{i}', f'transit_center_{i}', '-', NP(0.0, 0.1), (-inf, inf)) for i in range(self.nep)]
        else:
            ptc = [GParameter('tc', 'transit_center', '-', NP(0.0, 0.1), (-inf, inf))]

        self.ps.add_global_block('tc', ptc)
        self._start_tc = self.ps.blocks[-1].start
        self._sl_tc = self.ps.blocks[-1].slice

    def _init_p_planet(self):
        ps = self.ps
        pk2 = [PParameter('k2_true', 'apparent_area_ratio', 'A_s', UP(0.01**2, 0.30**2), (0., inf))]
        pcn = [GParameter('cnt_ref', 'reference_passband_contamination', '', UP(0.0, 1.0), bounds=(0.0, 1.0)),
               GParameter('teff_h', 'host_teff', 'K', UP(1200, 7000), bounds=(1200, 7000)),
               GParameter('teff_c', 'contaminant_teff', 'K', UP(1200, 7000), bounds=(1200, 7000)),
               GParameter('cnt_tess', 'tess_acontamination', '', UP(0.0, 1.), (0.0, 1.0))]
        ps.add_passband_block('k2', 1, 1, pk2)
        self._pid_k2 = repeat(ps.blocks[-1].start, self.npb)
        self._start_k2 = ps.blocks[-1].start
        self._sl_k2 = ps.blocks[-1].slice
        ps.add_global_block('contamination', pcn)
        self._pid_cn = arange(ps.blocks[-1].start, ps.blocks[-1].stop)
        self._start_cn = ps.blocks[-1].start
        self._sl_cn = ps.blocks[-1].slice

    def _init_lnlikelihood(self):
        if self.use_gp:
            self._add_lnlikelihood_model(CeleriteLogLikelihood(self, noise_ids=self.noise_ids[:self._stess]))
            self._add_lnlikelihood_model(WNLogLikelihood(self, noise_ids=self.noise_ids[self._stess:]))
        else:
            self._add_lnlikelihood_model(WNLogLikelihood(self))

    def _post_initialisation(self):
        if self.use_opencl:
            self.tm = self.tm.to_opencl()
        if self.model_ttvs:
            for i in range(self.nep):
                tc = self.zero_epoch + self.epochs[i]*self.period
                self.set_prior(f'tc_{i}', 'NP', tc.n, 5*tc.s)
        else:
            self.set_prior('tc', 'NP', self.zero_epoch.n, 2*self.zero_epoch.s)
        self.set_prior('p', 'NP', self.period.n, 2*self.period.s)
        self.set_prior('rho', 'UP', 1, 35)
        self.set_prior('k2_true', 'UP', round(0.5*self.toi.depth.n*1e-6, 5), round(1.5*self.toi.depth.n*1e-6, 5))
        self.set_prior('teff_h', 'NP', self.star.teff.n, self.star.teff.s)
        self.set_prior('teff_c', 'UP', 1200, 7000)
        if self.absolute_contamination:
            self.set_prior('cnt_ref', 'NP', 0.5, 1e-5)

    def optimize_global(self, niter: int = 200, npop: int = 100, population = None):
        if not self.absolute_contamination:
            p = self.ps[self._start_cn + 1].prior
            self.set_prior('teff_c', 'NP', p.mean, p.std)
            self.set_prior('cnt_ref', 'NP', 0.03, 0.0025)
        self.set_prior('cnt_tess', 'NP', 0.03, 0.0025)
        super().optimize_global(niter, npop, population=population)

    def sample_mcmc(self, niter: int = 500, thin: int = 5, repeats: int = 1, npop: int = None,
                    population=None, save: bool = False, set_teffc_prior: bool = True):
        if not self.absolute_contamination:
            if set_teffc_prior:
                self.set_prior('teff_c', 'UP', 1200.0, 7000.0)
            self.set_prior('cnt_ref', 'UP', 0.0, 1.0)
        self.set_prior('cnt_tess', 'UP', 0.0, 1.0)
        self.set_prior('k2_true', 'UP', 0.01 ** 2, 1.0)
        super().sample_mcmc(niter, thin, repeats, npop=npop, population=population, save=save)

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
        pvp = self.ps.sample_from_prior(npv)
        icn = self._start_cn
        if not self.absolute_contamination:
            pvp[:, icn] = uniform(0.0, 0.1, size=npv)
            pvp[:, icn + 2] = pvp[:, icn + 1]
        pvp[:, icn+3] = uniform(0.0, 0.1, size=npv)
        for p in self.ps[self._sl_lm]:
            if 'lm_i' in p.name:
                pvp[:, p.pid] = 0.01 * (pvp[:, p.pid] - 1.0) + 1.0
            else:
                pvp[:, p.pid] *= 0.01
        return pvp

    def transit_model(self, pvp):
        pvp = atleast_2d(pvp)
        cnt = zeros((pvp.shape[0], self.npb))
        zero_epoch = pvp[:,self._sl_tc] - self._tref
        period = pvp[:,0]
        smaxis = as_from_rhop(pvp[:, 1], period)
        inclination  = i_from_ba(pvp[:, 2], smaxis)
        radius_ratio = sqrt(pvp[:,self._sl_k2])
        ldc = pvp[:, self._sl_ld].reshape([-1, self.npb, 2])
        flux = self.tm.evaluate(radius_ratio, ldc, zero_epoch, period, smaxis, inclination)
        icn = self._start_cn
        teff_h = pvp[:, icn+1]
        teff_c = pvp[:, icn+2]
        cnt[:, 0] = pvp[:, icn+3]
        cnt[:, 1:] = self.cm.contamination(pvp[:, icn], teff_h, teff_c, absolute=self.absolute_contamination)
        if self.absolute_contamination:
            radius_p = poly1d([3.02018381e-04, -5.96908377e-01])
            area_ratio = (radius_p(teff_c) / radius_p(teff_h)) ** 2
            cnt[:, 1:] *= area_ratio[:,newaxis]
        return contaminate(flux, cnt, self.lcids, self.pbids)

    def posterior_samples(self, burn: int = 0, thin: int = 1, derived_parameters: bool = True):
        df = super().posterior_samples(burn, thin, False)
        if derived_parameters:
            df['a'] = as_from_rhop(df.rho.values, df.p.values)
            df['inc'] = i_from_baew(df.b.values, df.a.values, 0., 0.)
            df['k_true'] = sqrt(df.k2_true)
            df['k2_app'] = df.k2_true * (1 - df.cnt_ref)
            df['k_app'] = sqrt(df.k2_app)
            df['t14'] = d_from_pkaiews(df.p.values, df.k_true.values, df.a.values, df.inc.values, 0., 0., 1)
        return df

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

    def plot_gb_transits(self, solution: str = 'de', pv: ndarray = None, figsize: tuple = None, axes=None,
                         ncol: int = 4,
                         xlim: tuple = None, ylim: tuple = None, remove_baseline: bool = True, n_samples: int = 1500):

        solution = solution.lower()
        samples = None

        if pv is None:
            if solution == 'local':
                pv = self._local_minimization.x
            elif solution in ('de', 'global'):
                solution = 'global'
                pv = self.de.minimum_location
            elif solution in ('mcmc', 'mc'):
                solution = 'mcmc'
                samples = self.posterior_samples(derived_parameters=False)
                samples = permutation(samples.values)[:n_samples]
                pv = median(samples, 0)
            else:
                raise NotImplementedError("'solution' should be either 'local', 'global', or 'mcmc'")

        nlc = self.nlc - self._stess
        nrow = int(ceil(nlc / ncol))

        if axes is None:
            fig, axs = subplots(nrow, ncol, figsize=figsize, sharex='all', sharey='all', squeeze=False)
        else:
            fig, axs = None, axes

        [ax.autoscale(enable=True, axis='x', tight=True) for ax in axs.flat]

        if remove_baseline:
            if solution == 'mcmc':
                fbasel = median(self.baseline(samples), axis=0)
                fmodel, fmodm, fmodp = percentile(self.transit_model(samples), [50, 0.5, 99.5], axis=0)
            else:
                fbasel = squeeze(self.baseline(pv))
                fmodel, fmodm, fmodp = squeeze(self.transit_model(pv)), None, None
            fobs = self.ofluxa / fbasel
        else:
            if solution == 'mcmc':
                fbasel = median(self.baseline(samples), axis=0)
                fmodel, fmodm, fmodp = percentile(self.flux_model(samples), [50, 1, 99], axis=0)
            else:
                fbasel = squeeze(self.baseline(pv))
                fmodel, fmodm, fmodp = squeeze(self.flux_model(pv)), None, None
            fobs = self.ofluxa

        etess = self._stess

        for i in range(nlc):
            ax = axs.flat[i]
            sl = self.lcslices[etess + i]
            t = self.times[etess + i]
            if self.model_ttvs:
                e = self.epochs[self.tm.epids[etess + i]]
                tc = pv[self._start_tc + self.tm.epids[etess + i]]
            else:
                t0, p = pv[[self._start_tc, 0]]
                e = epoch(t.mean(), t0, p)
                tc = t0 + e * p

            tt = 24 * (t - tc)
            ax.plot(tt, fobs[sl], 'k.', alpha=0.2)
            ax.plot(tt, fmodel[sl], 'k')

            if solution == 'mcmc':
                ax.fill_between(tt, fmodm[sl], fmodp[sl], zorder=-100, alpha=0.2, fc='k')

            if not remove_baseline:
                ax.plot(tt, fbasel[sl], 'k--', alpha=0.2)

        setp(axs, xlim=xlim, ylim=ylim)
        setp(axs[-1, :], xlabel='Time - T$_c$ [h]')
        setp(axs[:, 0], ylabel='Normalised flux')

        if fig is not None:
            fig.tight_layout()
        return fig

    def plot_joint_marginals(self, figsize=None, nb=30, gs=25, with_contamination=False, **kwargs):
        df = self.posterior_samples(derived_parameters=True)
        if with_contamination:
            xlabels = ['$\Delta$ T$_\mathrm{Eff}$ [K]', 'Apparent radius ratio', 'Ref. pb. contamination',
                       'TESS contamination', 'Impact parameter', 'Stellar density [g/cm$^3$]']
            return _jplot([df.teff_c - df.teff_h, df.k_app, df.cnt_ref, df.cnt_tess, df.b, df.rho], df.k_true,
                          xlabels, 'True radius ratio', figsize, nb, gs, **kwargs)
        else:
            xlabels = ['$\Delta$ T$_\mathrm{Eff}$ [K]', 'Apparent radius ratio',
                       'Impact parameter', 'Stellar density [g/cm$^3$]']
            return _jplot([df.teff_c - df.teff_h, df.k_app, df.b, df.rho], df.k_true,
                          xlabels, 'True radius ratio', figsize, nb, gs, **kwargs)
