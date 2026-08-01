from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .parameters import ParametersManagerView
from .samples import SampleSetView
from .dial import DialCacheView


@dataclass(slots=True)
class PropagatorView:
    handle: Any

    def getParametersManager(self) -> ParametersManagerView:
        return ParametersManagerView(handle=self.handle.getParameterManager())

    def getSampleSet(self) -> SampleSetView:
        return SampleSetView(handle=self.handle.getSampleSet())

    def getDialCacheList(self) -> list[DialCacheView]:
        return [DialCacheView(elm) for elm in self.handle.getEventDialCache().getCache()]
