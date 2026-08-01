from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .samples import EventView

@dataclass(slots=True)
class DialView:
    handle: Any


@dataclass(slots=True)
class DialCacheView:
    handle: Any

    def getEvent(self) -> EventView:
        return EventView(self.handle.getEvent())

    def getDialList(self) -> list[DialView]:
        return [DialView(elm) for elm in self.handle.dialResponseCacheList()]
