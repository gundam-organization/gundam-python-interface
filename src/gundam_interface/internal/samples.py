from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class BinView:
    """Python-side view combining a GUNDAM histogram bin's data and context."""

    binContentHandle: Any
    binContextHandle: Any

    def getBin(self) -> Any:
        return self.binContextHandle.getBin()

    def getSumWeights(self) -> float:
        return float(self.binContentHandle.sumWeights)

    def setSumWeights(self, value: float) -> None:
        self.binContentHandle.sumWeights = float(value)

    def getSqrtSumSqWeights(self) -> float:
        return float(self.binContentHandle.sqrtSumSqWeights)

    def setSqrtSumSqWeights(self, value: float) -> None:
        self.binContentHandle.sqrtSumSqWeights = float(value)


@dataclass(slots=True)
class HistogramView:
    """Light Python-side view over a GUNDAM Histogram handle."""

    handle: Any

    def getNbBins(self) -> int:
        return int(self.handle.getNbBins())

    def getBinList(self) -> list[BinView]:
        binContents = list(self.handle.getBinContentList())
        binContexts = list(self.handle.getBinContextList())
        if len(binContents) != len(binContexts):
            raise ValueError(
                "GUNDAM histogram has inconsistent bin content and context lists: "
                f"{len(binContents)} != {len(binContexts)}"
            )
        return [
            BinView(
                binContentHandle=binContent,
                binContextHandle=binContext,
            )
            for binContent, binContext in zip(binContents, binContexts)
        ]


@dataclass(slots=True)
class EventView:
    """Light Python-side view over a GUNDAM Event handle."""

    handle: Any

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
        return EventView.Indices(self.handle.getIndices())

    def getWeights(self) -> EventView.Weights:
        return EventView.Weights(self.handle.getWeights())

    def getVariables(self) -> Any:
        return self.handle.getVariables()

    def getSize(self) -> int:
        return int(self.handle.getSize())

    def getEventWeight(self) -> float:
        return float(self.handle.getEventWeight())

    def getSummary(self, printVars: bool = True) -> str:
        return str(self.handle.getSummary(printVars))


@dataclass(slots=True)
class SampleView:
    """Light Python-side view over a GUNDAM Sample handle."""

    handle: Any

    def getName(self) -> str:
        return str(self.handle.getName())

    def getHistogram(self) -> HistogramView:
        return HistogramView(handle=self.handle.getHistogram())

    def getEventList(self) -> list[EventView]:
        return [
            EventView(handle=event)
            for event in self.handle.getEventList()
        ]


@dataclass(slots=True)
class SampleSetView:
    """Light Python-side view over a GUNDAM SampleSet handle."""

    handle: Any

    def getSampleList(self) -> list[SampleView]:
        return [
            SampleView(handle=sample)
            for sample in self.handle.getSampleList()
        ]
