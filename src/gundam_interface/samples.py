from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True, slots=True)
class GundamHistogram:
    """Python-side access to a GUNDAM histogram."""

    handle: Any

    @property
    def binContents(self) -> list[Any]:
        """Raw GUNDAM bin-content handles."""
        return list(self.handle.getBinContentList())

    @property
    def sumWeights(self) -> np.ndarray:
        """Bin ``sumWeights`` values as a NumPy array."""
        return np.array(
            [float(binContent.sumWeights) for binContent in self.binContents],
            dtype=np.float64,
        )

    @property
    def sqrtSumSqWeights(self) -> np.ndarray:
        """Bin ``sqrtSumSqWeights`` values as a NumPy array."""
        return np.array(
            [float(binContent.sqrtSumSqWeights) for binContent in self.binContents],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class GundamSample:
    """Python-side access to a GUNDAM sample."""

    index: int
    handle: Any

    @property
    def histogram(self) -> GundamHistogram:
        """Sample histogram."""
        return GundamHistogram(self.handle.getHistogram())

    @property
    def sumWeights(self) -> np.ndarray:
        """Histogram bin ``sumWeights`` values."""
        return self.histogram.sumWeights


@dataclass(frozen=True, slots=True)
class GundamSamples:
    """Collection wrapper for GUNDAM samples."""

    propagator: Any

    @property
    def sampleSet(self) -> Any:
        """Raw GUNDAM sample set handle."""
        return self.propagator.getSampleSet()

    @property
    def handles(self) -> list[Any]:
        """Raw GUNDAM sample handles."""
        return list(self.sampleSet.getSampleList())

    def __len__(self) -> int:
        return len(self.handles)

    def __iter__(self) -> Iterator[GundamSample]:
        for index, sample in enumerate(self.handles):
            yield GundamSample(index=index, handle=sample)

    def __getitem__(self, index: int) -> GundamSample:
        return GundamSample(index=index, handle=self.handles[index])

    def histogram(self, index: int) -> GundamHistogram:
        """Histogram for one sample."""
        return self[index].histogram

    def sumWeights(self, index: int) -> np.ndarray:
        """Histogram bin ``sumWeights`` values for one sample."""
        return self[index].sumWeights

    def allSumWeights(self) -> list[np.ndarray]:
        """Histogram bin ``sumWeights`` values for every sample."""
        return [sample.sumWeights for sample in self]
