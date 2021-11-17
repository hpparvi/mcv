from astropy.io import fits as pf
from astropy.stats import sigma_clip
from astropy.table import Table
from numpy import isfinite, array, arange
from pytransit.utils import downsample_time_1d, downsample_time_2d

from mcv.io.photometry import Photometry


def read_m2_data(files, downsample=None, passbands=('g', 'r', 'i', 'z_s'), heavy_baseline: bool = True):
    times, fluxes, pbs, wns, covs = [], [], [], [], []
    for inight, f in enumerate(files):
        with pf.open(f) as hdul:
            npb = (len(hdul)-1)//2
            for ipb in range(npb):
                hdu = hdul[1 + ipb]
                pb = hdu.header['filter']
                if pb in passbands:
                    fobs = hdu.data['flux'].astype('d').copy()
                    fmod = hdu.data['model'].astype('d').copy()
                    time = hdu.data['time_bjd'].astype('d').copy()
                    mask = ~sigma_clip(fobs-fmod, sigma=5).mask

                    wns.append(hdu.header['wn'])
                    pbs.append(pb)

                    if downsample is None:
                        times.append(time[mask])
                        fluxes.append(fobs[mask])
                        covs.append(Table.read(f, 1+npb+ipb).to_pandas().values[mask,1:])
                    else:
                        cov = Table.read(f, 1+npb+ipb).to_pandas().values[mask,1:]
                        tb, fb, eb = downsample_time_1d(time[mask], fobs[mask], downsample / 24 / 60)
                        _,  cb, _ = downsample_time_2d(time[mask], cov, downsample / 24 / 60)
                        m = isfinite(tb)
                        times.append(tb[m])
                        fluxes.append(fb[m])
                        covs.append(cb[m])
    if not heavy_baseline:
        covs = len(times) * [array([[]])]
    ins = len(times)*["M2"]
    piis = list(arange(len(times)))
    exptimes = len(times)*[0.0]
    nsamples = len(times)*[1]
    return Photometry(times, fluxes, covs, pbs, wns, ins, piis, exptimes, nsamples)