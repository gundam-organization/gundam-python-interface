from __future__ import annotations

from dataclasses import dataclass
from operator import index as integerIndex
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
class VariableCollectionView:
    """Light Python-side view over a GUNDAM VariableCollection handle."""

    handle: Any

    def __getitem__(self, key: str | int) -> int | float:
        return self._getVariableHolder(key).getValue()

    def __setitem__(self, key: str | int, value: float) -> None:
        """Update a stored scalar, preserving its C++ type and cached value."""
        self._getVariableHolder(key).setValue(value)

    def _getVariableHolder(self, key: str | int) -> Any:
        if isinstance(key, str):
            idx = self.handle.findVarIndex(key, False)
            if idx == -1:
                raise KeyError(key)
        else:
            idx = integerIndex(key)
        return self.handle.getVarList()[idx]

    def toDict(self) -> dict[str, int | float]:
        """Return variable names and their current values as a dictionary."""
        return {
            name: self[idx]
            for idx, name in enumerate(self.handle.getNameList())
        }

    def __str__(self) -> str:
        return str(self.toDict())



@dataclass(slots=True)
class EventView:
    """Light Python-side view over a GUNDAM Event handle."""

    handle: Any

    @dataclass(frozen=True, slots=True)
    class Indices:
        """Python value object for the source indices of a GUNDAM event."""
        handle: Any

        @property
        def dataset(self) -> int:
            return int(self.handle.dataset)

        @property
        def sample(self) -> int:
            return int(self.handle.sample)

        @property
        def bin(self) -> int:
            return int(self.handle.bin)

        @property
        def entry(self) -> int:
            return int(self.handle.entry)

        @property
        def treeFile(self) -> int:
            return int(self.handle.treeFile)

        @property
        def treeEntry(self) -> int:
            return int(self.handle.treeEntry)


    @dataclass(frozen=True, slots=True)
    class Weights:
        """Python value object for the weights of a GUNDAM event."""
        handle: Any

        @property
        def base(self) -> float:
            return float(self.handle.base)

        @property
        def current(self) -> float:
            return float(self.handle.current)


    def getIndices(self) -> EventView.Indices:
        return EventView.Indices(self.handle.getIndices())

    def getWeights(self) -> EventView.Weights:
        return EventView.Weights(self.handle.getWeights())

    def getVariables(self) -> VariableCollectionView:
        return VariableCollectionView(self.handle.getVariables())

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

    def setUseDynamicBinning(self, useDynamicBinning: bool) -> None:
        """Enable before initialization to retain unbinned events and binning variables."""
        self.handle.setUseDynamicBinning(useDynamicBinning)

    def isUseDynamicBinning(self) -> bool:
        return bool(self.handle.isUseDynamicBinning())

    def updateEventBinning(self) -> None:
        """Reassign this sample's events and rebuild its bin event lists.

        Only samples with dynamic binning enabled are updated. For repeated
        updates, prefer PropagatorView.updateEventBinning(), which reuses its pool.
        For model samples, evaluateLlh() afterwards refills the histogram.
        Event-to-dial associations are preserved.
        """
        self.handle.updateEventBinning()

    def getNbBinnedEvents(self) -> int:
        return int(self.handle.getNbBinnedEvents())

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
