from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class HistogramView:
    """Light Python-side view over a GUNDAM Histogram handle."""

    _handle: Any

    def getNbBins(self) -> int:
        return int(self._handle.getNbBins())

    def getBinContentList(self) -> list[Any]:
        return list(self._handle.getBinContentList())

    def getBinContextList(self) -> list[Any]:
        return list(self._handle.getBinContextList())


@dataclass(slots=True)
class SampleView:
    """Light Python-side view over a GUNDAM Sample handle."""

    _handle: Any

    def getName(self) -> str:
        return str(self._handle.getName())

    def getHistogram(self) -> HistogramView:
        return HistogramView(_handle=self._handle.getHistogram())


@dataclass(slots=True)
class SampleSetView:
    """Light Python-side view over a GUNDAM SampleSet handle."""

    _handle: Any

    def getSampleList(self) -> list[SampleView]:
        return [
            SampleView(_handle=sample)
            for sample in self._handle.getSampleList()
        ]
