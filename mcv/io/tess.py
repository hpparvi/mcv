from copy import copy

from astropy.table import Table
from numpy import nanmedian, zeros, arange, array, diff, concatenate, sqrt
from pytransit.utils.keplerlc import KeplerLC
from uncertainties import nominal_value

from mcv.io.photometry import Photometry


def read_tess_data(dfiles, zero_epoch: float, period: float, use_pdc: bool = False,
              transit_duration_d: float = 0.1, baseline_duration_d: float = 0.3):
    times, fluxes, ins, piis, exptimes, nsamples = [], [], [], [], [], []
    for dfile in dfiles:
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
        m = flux > 0.9
        lc = KeplerLC(time[m], flux[m], zeros(flux[m].size), nominal_value(zero_epoch), nominal_value(period), transit_duration_d, baseline_duration_d)
        times.extend(copy(lc.time_per_transit))
        cfluxes = copy(lc.normalized_flux_per_transit)
        if use_pdc and 'CROWDSAP' in tb.meta:
            contamination = 1 - tb.meta['CROWDSAP']
            cfluxes = [contamination + (1 - contamination) * f for f in cfluxes]
        fluxes.extend(cfluxes)
        exptimes.extend(len(cfluxes)*[0.0 if source == 'SPOC' else 0.021])
        nsamples.extend(len(cfluxes)*[1 if source == 'SPOC' else 10])

    ins = len(times) * ["TESS"]
    piis = list(arange(len(times)))
    return Photometry(times, fluxes, len(times) * [array([[]])], len(times) * ['tess'], [diff(concatenate(fluxes)).std() / sqrt(2)],
                      ins, piis, exptimes, nsamples)