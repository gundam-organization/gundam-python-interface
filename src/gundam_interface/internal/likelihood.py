from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dial import EventDialCacheView
from .parameters import ParametersManagerView
from .propagator import PropagatorView
from .samples import SampleSetView


@dataclass(slots=True)
class LikelihoodInterfaceView:
    handle: Any

    def getSummary(self) -> str:
        return str(self.handle.getSummary())

    def getSampleBreakdownTable(self) -> str:
        return str(self.handle.getSampleBreakdownTable())

    def getLastLikelihood(self) -> float:
        return float(self.handle.getLastLikelihood())

    def getLastStatLikelihood(self) -> float:
        return float(self.handle.getLastStatLikelihood())

    def getLastPenaltyLikelihood(self) -> float:
        return float(self.handle.getLastPenaltyLikelihood())

    def propagateAndEvalLikelihood(self):
        self.handle.propagateAndEvalLikelihood()

    def throwStatErrors(self, propagator_: PropagatorView):
        self.handle.throwStatErrors(propagator_.handle)

    def throwToyParameters(self, propagator_: PropagatorView):
        self.handle.throwToyParameters(propagator_.handle)

    def setCurrentParameterValuesAsPrior(self):
        self.handle.setCurrentParameterValuesAsPrior()
