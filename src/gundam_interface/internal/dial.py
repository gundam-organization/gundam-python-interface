from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .samples import EventView
from .parameters import ParameterView


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
        return float(self.handle.getResponse())


@dataclass(slots=True)
class EventDialCacheView:
    """Light Python-side view over a GUNDAM CacheEntry"""
    handle: Any

    def getEvent(self) -> EventView:
        return EventView(self.handle.getEvent())

    def getDialList(self) -> list[DialView]:
        return [DialView(elm) for elm in self.handle.dialResponseCacheList()]

    def getDialListAffecting(self, parameter_: ParameterView) -> list[DialView]:
        return [DialView(elm) for elm in self.handle.getDialListAffecting(parameter_.handle)]
