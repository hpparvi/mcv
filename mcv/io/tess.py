from copy import copy
from pathlib import Path
from typing import Union, Optional, List

from astropy.io.fits import getval
from astropy.stats import mad_std
from astropy.table import Table
from numpy import nanmedian, zeros, arange, array, diff, concatenate, sqrt, ones, inf, median, isfinite
from pytransit.orbits import fold
from pytransit.utils.keplerlc import KeplerLC
from uncertainties import nominal_value

from .photometry import Photometry


def identify_tess_format_and_sector(f: Path):
    f = Path(f)
    if 'hlsp_qlp' in f.name:
        fmt = 'QLP'
        sector = int(f.name.split('_')[4].split('-')[0][1:])
    elif 'hlsp_tess-spoc' in f.name:
        fmt = 'TESS-SPOC'
        sector = int(f.name.split('_')[4].split('-')[1][1:])
    elif f.name[:6] == 'tess20':
        fmt = 'SPOC'
        sector = int(f.name.split('-')[1][1:])
    else:
        fmt = None
        sector = None
    return f, fmt, sector


def get_tess_files(d: Path):
    files = [identify_tess_format_and_sector(f) for f in Path(d).glob('*.fits')]
    return sorted(files, key=lambda f:f[2])


def read_qlp(f: Path, use_pdc: bool = True):
    tb = Table.read(f)
    m = (tb['QUALITY'] == 0) & isfinite(tb['TIME'])
    time = tb['TIME'].data[m] + tb.meta['BJDREFI']
    flux = tb['KSPSAP_FLUX'].data[m].astype('d') if use_pdc else tb['SAP_FLUX'].data[m].astype('d')
    return time, flux / median(flux), tb.meta['TIMEDEL']


def read_tess_spoc(f: Path, use_pdc: bool = True):
    tb = Table.read(f)
    m = (tb['QUALITY'] == 0) & isfinite(tb['TIME'])
    time = array(tb['TIME'].data[m] + tb.meta['BJDREFI'])
    flux = array(tb['PDCSAP_FLUX'].data[m].astype('d') if use_pdc else tb['SAP_FLUX'].data[m].astype('d'))
    flux /= median(flux)
    if use_pdc:
        contamination = 1 - tb.meta['CROWDSAP']
        flux = contamination + (1 - contamination) * flux
    return time, flux, tb.meta['TIMEDEL']


def read_spoc(f: Path, use_pdc: bool = True):
    tb = Table.read(f)
    m = (tb['QUALITY'] == 0) & isfinite(tb['TIME'])
    time = array(tb['TIME'].data[m] + tb.meta['BJDREFI'])
    flux = array(tb['PDCSAP_FLUX'].data[m].astype('d') if use_pdc else tb['SAP_FLUX'].data[m].astype('d'))
    flux /= median(flux)
    if use_pdc:
        contamination = 1 - tb.meta['CROWDSAP']
        flux = contamination + (1 - contamination) * flux
    return time, flux, tb.meta['TIMEDEL']


tess_readers = {'SPOC': read_spoc, 'TESS-SPOC': read_tess_spoc, 'QLP': read_qlp}


def read_tess(tic: int, datadir: Union[Path, str],
              sectors: Optional[Union[List[int], str]] = 'all',
              use_pdc: bool = True,  split_transits: bool = False,
              zero_epoch: Optional[float] = None, period: Optional[float] = None,
              transit_duration: float = 0.1, baseline_duration: float = 0.3,
              depth: float = 0.0, sigma_low: float = 5.0, sigma_high: float= 5.0):

    files = get_tess_files(datadir)

    times, fluxes, sectors, exptimes, nsamples = [], [], [], [], []
    for (fname, fmt, sector) in files:
        time, flux, exptime = tess_readers[fmt](fname, use_pdc)
        wn = mad_std(diff(flux)) / sqrt(2)

        m = (flux >= 1 - depth - sigma_low*wn) & (flux <= 1 + sigma_high*wn)
        if zero_epoch and period:
            phase = fold(time, period, zero_epoch)
            m &= abs(phase) <= 0.5*baseline_duration

        if split_transits :
            if zero_epoch is None or period is None:
                raise ValueError('Both zero_epoch and period must be given if split_transits == True')
            lc = KeplerLC(time[m], flux[m], zeros(flux[m].size), nominal_value(zero_epoch), nominal_value(period), transit_duration, baseline_duration)
            times.extend(copy(lc.time_per_transit))
            cfluxes = copy(lc.normalized_flux_per_transit)
        else:
            times.append(time[m])
            cfluxes = [flux[m]]

        fluxes.extend(cfluxes)
        sectors.extend(len(cfluxes)*[sector])
        exptimes.extend(len(cfluxes)*[exptime])
        nsamples.extend(len(cfluxes)*[max(1, int(exptime / 0.0013889))])

    instrument = len(times) * ["TESS"]
    segments = list(arange(len(times)))
    return Photometry(times, fluxes, len(times) * [array([[]])], len(times) * ['tess'], [diff(concatenate(fluxes)).std() / sqrt(2)],
                      instrument, sectors, segments, exptimes, nsamples)