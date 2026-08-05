from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .parameters import ParameterView
from .samples import EventView


@dataclass(slots=True, init=False)
class DialView:
    """Light Python-side view over a GUNDAM DialResponseCache (multiple class merged)"""
    handle: Any

    def __init__(self, handle: Any) -> None:
        self.handle = handle

    def getCachedResponse(self) -> float:
        return float(self.handle.response)

    def getDialSummary(self) -> str:
        return str(self.handle.getDialInterface().getSummary())

    def evaluateResponse(self) -> float:
        # we need to update the intput buffer manually
        self.updateInputBuffer()
        # return the response
        return float(self.handle.getDialInterface().evalResponse())

    def updateInputBuffer(self) -> None:
        self.handle.getDialInterface().getInputBuffer().update()


@dataclass(slots=True)
class EventDialCacheView:
    """Light Python-side view over a GUNDAM CacheEntry"""
    handle: Any

    def getEvent(self) -> EventView:
        return EventView(self.handle.getEvent())

    def getDialList(self) -> list[DialView]:
        return [DialView(elm) for elm in self.handle.dialResponseCacheList()]

    def getDialListAffecting(self, parameter_: ParameterView) -> list[DialView]:
        idxList = list(self.handle.getDialIndicesAffecting(parameter_.handle))
        return [DialView(self.handle.dialResponseCacheList[idx]) for idx in idxList]
