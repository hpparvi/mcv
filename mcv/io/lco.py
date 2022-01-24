from pathlib import Path
from typing import Union

from pandas import read_csv
from numpy import diff, sqrt, sum

from .photometry import Photometry

def read_lco(fname: Union[Path, str], passband: str, instrument: str, delimiter: str = 'whitespace',
             time_col: str = 'BJD_TDB', flux_col: str = 'rel_flux_T1',
             cov_cols: str = 'AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'):

    if delimiter == 'whitespace':
        df = read_csv(fname, delim_whitespace=True)
    else:
        df = read_csv(fname, delimiter=delimiter)

    time = [df[time_col].values.copy()]
    flux = [df[flux_col].values.copy()]
    cov = [df[cov_cols.split()].values.copy()]
    wn = [diff(flux[0]).std() / sqrt(2)]
    return Photometry(time, flux, cov, [passband], wn, [instrument], [0], [0], [0.0], [1])

def read_m3(datadir: Path, heavy_baseline: bool = False):
    def sort_key(s):
        d = dict(gp=0.0, rp=0.1, ip=0.2, zs=0.3)
        _, date, _, pb, _ = s.name.split('_')
        return float(date) + d[pb]
    files = sorted(Path(datadir).glob('*HAL-2m0*'), key=sort_key)
    cov_cols = 'AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1' if heavy_baseline else ''
    get_pb = lambda s: s.name.split('_')[3][0]
    data = sum([read_lco(f, passband=get_pb(f), instrument='M3', cov_cols=cov_cols) for f in files])
    data.passband = [pb.replace('z', 'z_s') for pb in data.passband]
    return data