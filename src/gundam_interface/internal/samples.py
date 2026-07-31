from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class HistogramBinView:
    """Python-side view combining a GUNDAM histogram bin's data and context."""

    _binContentHandle: Any
    _binContextHandle: Any

    def getBin(self) -> Any:
        return self._binContextHandle.getBin()

    def getSumWeights(self) -> float:
        return float(self._binContentHandle.sumWeights)

    def setSumWeights(self, value: float) -> None:
        self._binContentHandle.sumWeights = float(value)

    def getSqrtSumSqWeights(self) -> float:
        return float(self._binContentHandle.sqrtSumSqWeights)

    def setSqrtSumSqWeights(self, value: float) -> None:
        self._binContentHandle.sqrtSumSqWeights = float(value)


@dataclass(slots=True)
class HistogramView:
    """Light Python-side view over a GUNDAM Histogram handle."""

    _handle: Any

    def getNbBins(self) -> int:
        return int(self._handle.getNbBins())

    def getBinList(self) -> list[HistogramBinView]:
        binContents = list(self._handle.getBinContentList())
        binContexts = list(self._handle.getBinContextList())
        if len(binContents) != len(binContexts):
            raise ValueError(
                "GUNDAM histogram has inconsistent bin content and context lists: "
                f"{len(binContents)} != {len(binContexts)}"
            )
        return [
            HistogramBinView(
                _binContentHandle=binContent,
                _binContextHandle=binContext,
            )
            for binContent, binContext in zip(binContents, binContexts)
        ]


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
