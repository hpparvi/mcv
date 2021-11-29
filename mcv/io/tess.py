from copy import copy
from pathlib import Path
from typing import Union, Optional, List

from astropy.io.fits import getval
from astropy.stats import mad_std
from astropy.table import Table
from numpy import nanmedian, zeros, arange, array, diff, concatenate, sqrt, ones, inf
from pytransit.orbits import fold
from pytransit.utils.keplerlc import KeplerLC
from uncertainties import nominal_value

from .photometry import Photometry

def file_filter(f, partial_tic, sectors):
    if 'qlp' in f.name:
        sector, tic = f.name.split('_')[4].split('-')
    else:
        _, sector, tic, _, _ = f.name.split('-')

    if sectors != 'all':
        return int(sector[1:]) in sectors and str(partial_tic) in tic
    else:
        return str(partial_tic) in tic

def read_tess(tic: int, datadir: Union[Path, str],
              sectors: Optional[Union[List[int], str]] = 'all',
              use_pdc: bool = True,  split_transits: bool = False,
              zero_epoch: Optional[float] = None, period: Optional[float] = None,
              transit_duration: float = 0.1, baseline_duration: float = 0.3,
              depth: float = 0.0, sigma_low: float = 5.0, sigma_high: float= 5.0):

    files = [f for f in sorted(Path(datadir).glob('*tess*lc.fits')) if file_filter(f, tic, sectors)]

    times, fluxes, sectors, exptimes, nsamples = [], [], [], [], []
    for dfile in files:
        tb = Table.read(dfile)

        if 'PDCSAP_FLUX' in tb.colnames:
            source = 'SPOC'
            fcol = 'PDCSAP_FLUX' if use_pdc else 'SAP_FLUX'
        elif 'KSPSAP_FLUX' in tb.colnames:
            source = 'QLP'
            fcol = 'KSPSAP_FLUX'

        bjdrefi = tb.meta['BJDREFI']
        df = tb.to_pandas().dropna(subset=['TIME', fcol])
        time = df.TIME.values + bjdrefi
        flux = df[fcol].values
        flux /= nanmedian(flux)

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

        if use_pdc and 'CROWDSAP' in tb.meta:
            contamination = 1 - tb.meta['CROWDSAP']
            cfluxes = [contamination + (1 - contamination) * f for f in cfluxes]
        fluxes.extend(cfluxes)
        sectors.extend(len(cfluxes)*[getval(dfile, 'sector')])
        exptimes.extend(len(cfluxes)*[0.0 if source == 'SPOC' else 0.021])
        nsamples.extend(len(cfluxes)*[1 if source == 'SPOC' else 10])

    instrument = len(times) * ["TESS"]
    segments = list(arange(len(times)))
    return Photometry(times, fluxes, len(times) * [array([[]])], len(times) * ['tess'], [diff(concatenate(fluxes)).std() / sqrt(2)],
                      instrument, sectors, segments, exptimes, nsamples)