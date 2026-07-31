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
class EventView:
    """Light Python-side view over a GUNDAM Event handle."""

    _handle: Any

    @dataclass(frozen=True, slots=True, init=False)
    class Indices:
        """Python value object for the source indices of a GUNDAM event."""

        dataset: int
        treeFile: int
        sample: int
        bin: int
        entry: int
        treeEntry: int

        def __init__(self, handle: Any) -> None:
            object.__setattr__(self, "dataset", int(handle.dataset))
            object.__setattr__(self, "treeFile", int(handle.treeFile))
            object.__setattr__(self, "sample", int(handle.sample))
            object.__setattr__(self, "bin", int(handle.bin))
            object.__setattr__(self, "entry", int(handle.entry))
            object.__setattr__(self, "treeEntry", int(handle.treeEntry))

    @dataclass(frozen=True, slots=True, init=False)
    class Weights:
        """Python value object for the weights of a GUNDAM event."""

        base: float
        current: float

        def __init__(self, handle: Any) -> None:
            object.__setattr__(self, "base", float(handle.base))
            object.__setattr__(self, "current", float(handle.current))

    def getIndices(self) -> EventView.Indices:
        return EventView.Indices(self._handle.getIndices())

    def getWeights(self) -> EventView.Weights:
        return EventView.Weights(self._handle.getWeights())

    def getVariables(self) -> Any:
        return self._handle.getVariables()

    def getSize(self) -> int:
        return int(self._handle.getSize())

    def getEventWeight(self) -> float:
        return float(self._handle.getEventWeight())

    def getSummary(self, printVars: bool = True) -> str:
        return str(self._handle.getSummary(printVars))


@dataclass(slots=True)
class SampleView:
    """Light Python-side view over a GUNDAM Sample handle."""

    _handle: Any

    def getName(self) -> str:
        return str(self._handle.getName())

    def getHistogram(self) -> HistogramView:
        return HistogramView(_handle=self._handle.getHistogram())

    def getEventList(self) -> list[EventView]:
        return [
            EventView(_handle=event)
            for event in self._handle.getEventList()
        ]


@dataclass(slots=True)
class SampleSetView:
    """Light Python-side view over a GUNDAM SampleSet handle."""

    _handle: Any

    def getSampleList(self) -> list[SampleView]:
        return [
            SampleView(_handle=sample)
            for sample in self._handle.getSampleList()
        ]
