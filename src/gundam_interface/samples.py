from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class GundamHistogram:
    """Python-side access to a GUNDAM histogram."""

    handle: Any

    @staticmethod
    def _chooseVariableOrder(
        discoveredNames: Sequence[str], preferredOrder: Sequence[str] | None = None
    ) -> list[str]:
        discoveredNames = list(discoveredNames)
        if preferredOrder is None:
            return discoveredNames

        ordered = [name for name in preferredOrder if name in discoveredNames]
        ordered.extend(name for name in discoveredNames if name not in ordered)
        if len(ordered) != len(discoveredNames):
            raise ValueError(
                f"Inconsistent axis order: {ordered} vs {discoveredNames}"
            )
        return ordered

    @property
    def binContents(self) -> list[Any]:
        """Raw GUNDAM bin-content handles."""
        return list(self.handle.getBinContentList())

    @property
    def binContexts(self) -> list[Any]:
        """Raw GUNDAM bin-context handles."""
        return list(self.handle.getBinContextList())

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

    def variableNames(
        self,
        preferredOrder: Sequence[str] | None = None,
        *,
        includeConditionVars: bool = False,
    ) -> list[str]:
        """Variable names used by this histogram."""
        discoveredNames = None
        for binContext in self.binContexts:
            edges = [
                edge
                for edge in binContext.bin.getEdgesList()
                if includeConditionVars or not edge.isConditionVar
            ]
            currentNames = [edge.varName for edge in edges]
            if discoveredNames is None:
                discoveredNames = currentNames
                continue
            if set(currentNames) != set(discoveredNames):
                raise ValueError(
                    "Inconsistent variables across bins: "
                    f"{currentNames} vs {discoveredNames}"
                )

        if discoveredNames is None:
            return []

        return self._chooseVariableOrder(discoveredNames, preferredOrder)

    def binDefinitions(
        self,
        variableOrder: Sequence[str] | None = None,
        *,
        includeConditionVars: bool = False,
    ) -> list[dict[str, Any]]:
        """Bin definitions indexed like the histogram contents."""
        if variableOrder is None:
            variableOrder = self.variableNames(
                includeConditionVars=includeConditionVars
            )
        else:
            variableOrder = list(variableOrder)

        bins = []
        nBinContents = len(self.binContents)
        for binContext in self.binContexts:
            binHandle = binContext.bin
            edges = [
                edge
                for edge in binHandle.getEdgesList()
                if includeConditionVars or not edge.isConditionVar
            ]
            edgeByName = {edge.varName: edge for edge in edges}
            if set(edgeByName) != set(variableOrder):
                raise ValueError(
                    f"Inconsistent variables for bin {binHandle.getIndex()}: "
                    f"{list(edgeByName)} vs {list(variableOrder)}"
                )

            binIndex = int(binHandle.getIndex())
            if not 0 <= binIndex < nBinContents:
                raise IndexError(
                    f"Invalid bin index {binIndex} for {nBinContents} contents"
                )

            bins.append(
                {
                    "index": binIndex,
                    "edges": {
                        name: {
                            "min": float(edgeByName[name].min),
                            "max": float(edgeByName[name].max),
                        }
                        for name in variableOrder
                    },
                }
            )

        bins.sort(key=lambda item: item["index"])
        return bins

    def layout2d(
        self, preferredOrder: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """2D bin layout and contents for runtime-defined histogram plotting."""
        variableNames = self.variableNames(preferredOrder=preferredOrder)
        if len(variableNames) != 2:
            raise ValueError(
                f"Expected a 2D histogram, got {len(variableNames)} dimensions"
            )

        binDefinitions = self.binDefinitions(variableOrder=variableNames)
        sumWeights = np.array(self.sumWeights, copy=True)

        xName, yName = variableNames
        return {
            "variable_names": variableNames,
            "bins": [
                {
                    "index": binDefinition["index"],
                    "x_min": binDefinition["edges"][xName]["min"],
                    "x_max": binDefinition["edges"][xName]["max"],
                    "y_min": binDefinition["edges"][yName]["min"],
                    "y_max": binDefinition["edges"][yName]["max"],
                }
                for binDefinition in binDefinitions
            ],
            "sum_weights": sumWeights,
            "x_edges": np.unique(
                [
                    edge
                    for binDefinition in binDefinitions
                    for edge in (
                        binDefinition["edges"][xName]["min"],
                        binDefinition["edges"][xName]["max"],
                    )
                ]
            ),
            "y_edges": np.unique(
                [
                    edge
                    for binDefinition in binDefinitions
                    for edge in (
                        binDefinition["edges"][yName]["min"],
                        binDefinition["edges"][yName]["max"],
                    )
                ]
            ),
        }


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
