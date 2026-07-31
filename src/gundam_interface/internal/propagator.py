from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .parameters import ParametersManagerView
from .samples import SampleSetView


@dataclass(slots=True)
class PropagatorView:
    _handle: Any

    def getParametersManager(self) -> ParametersManagerView:
        return ParametersManagerView(self._handle.getParameterManager())

    def getSampleSet(self) -> SampleSetView:
        return SampleSetView(self._handle.getSampleSet())