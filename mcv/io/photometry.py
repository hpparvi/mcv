from dataclasses import dataclass
from typing import List


@dataclass
class Photometry:
    times: List
    fluxes: List
    covariates: List
    passbands: List
    noise: List
    instrument: List
    piis: List
    exptimes: List
    nsamples: List

    def __add__(self, other):
        return Photometry(self.times+other.times, self.fluxes+other.fluxes, self.covariates+other.covariates,
                          self.passbands+other.passbands, self.noise+other.noise, self.instrument+other.instrument,
                          self.piis+other.piis, self.exptimes+other.exptimes, self.nsamples+other.nsamples)