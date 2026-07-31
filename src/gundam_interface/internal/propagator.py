from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .parameters import ParametersManagerView
from .samples import SampleSetView
from .engine import EngineView


@dataclass(slots=True)
class PropagatorView:
    handle: Any

    def getParametersManager(self) -> ParametersManagerView:
        return ParametersManagerView(handle=self.handle.getParameterManager())

    def getSampleSet(self) -> SampleSetView:
        return SampleSetView(handle=self.handle.getSampleSet())

    def getEngine(self) -> EngineView:
        return EngineView(handle=self.handle.getEventDialCache().getCache())
