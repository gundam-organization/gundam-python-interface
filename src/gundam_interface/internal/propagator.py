from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .parameters import ParametersManagerView
from .samples import SampleSetView
from .dial import EventDialCacheView


@dataclass(slots=True)
class PropagatorView:
    handle: Any

    def getParametersManager(self) -> ParametersManagerView:
        return ParametersManagerView(handle=self.handle.getParametersManager())

    def getSampleSet(self) -> SampleSetView:
        return SampleSetView(handle=self.handle.getSampleSet())

    def getDialCacheList(self) -> list[EventDialCacheView]:
        return [EventDialCacheView(elm) for elm in self.handle.getEventDialCache().getCache()]
