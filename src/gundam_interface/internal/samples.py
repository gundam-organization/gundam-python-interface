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

    @property
    def binVolumes(self) -> np.ndarray:
        """Full-dimensional bin volumes as a NumPy array."""
        return self.binMeasures()

    def densitySumWeights(self) -> np.ndarray:
        """Bin ``sumWeights`` values divided by the full-dimensional bin volume."""
        return np.divide(
            self.sumWeights,
            self.binVolumes,
            out=np.full_like(self.sumWeights, np.nan),
            where=self.binVolumes != 0,
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
            missingNames = [name for name in variableOrder if name not in edgeByName]
            if missingNames:
                raise ValueError(
                    f"Inconsistent variables for bin {binHandle.getIndex()}: "
                    f"missing {missingNames} in {list(edgeByName)}"
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

    def binMeasures(
        self,
        variableOrder: Sequence[str] | None = None,
        *,
        includeConditionVars: bool = False,
    ) -> np.ndarray:
        """Per-bin measure computed from the selected variables only."""
        if variableOrder is None:
            variableOrder = self.variableNames(
                includeConditionVars=includeConditionVars
            )
        else:
            variableOrder = list(variableOrder)

        binDefinitions = self.binDefinitions(
            variableOrder=variableOrder,
            includeConditionVars=includeConditionVars,
        )
        measures = np.full(len(binDefinitions), np.nan, dtype=np.float64)
        for binDefinition in binDefinitions:
            measure = 1.0
            for edges in binDefinition["edges"].values():
                measure *= float(edges["max"]) - float(edges["min"])
            measures[binDefinition["index"]] = measure
        return measures

    def projectedDensitySumWeights(
        self,
        variableOrder: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Bin ``sumWeights`` values divided by the selected-variable measure."""
        measures = self.binMeasures(variableOrder=variableOrder)
        return np.divide(
            self.sumWeights,
            measures,
            out=np.full_like(self.sumWeights, np.nan),
            where=measures != 0,
        )

    def layout2d(
        self,
        preferredOrder: Sequence[str] | None = None,
        *,
        divideByBinVolume: bool = False,
    ) -> dict[str, Any]:
        """2D bin layout and contents for runtime-defined histogram plotting."""
        allVariableNames = self.variableNames()
        variableNames = self._chooseVariableOrder(allVariableNames, preferredOrder)[:2]
        if len(variableNames) != 2:
            raise ValueError(
                f"Expected at least 2 dimensions, got {len(allVariableNames)}"
            )

        xName, yName = variableNames
        binDefinitions = self.binDefinitions(variableOrder=allVariableNames)
        projectedBins = {}
        for binDefinition, sumWeight in zip(binDefinitions, self.sumWeights):
            xEdges = binDefinition["edges"][xName]
            yEdges = binDefinition["edges"][yName]
            key = (
                float(xEdges["min"]),
                float(xEdges["max"]),
                float(yEdges["min"]),
                float(yEdges["max"]),
            )
            if key not in projectedBins:
                projectedBins[key] = {
                    "x_min": key[0],
                    "x_max": key[1],
                    "y_min": key[2],
                    "y_max": key[3],
                    "measure": (key[1] - key[0]) * (key[3] - key[2]),
                    "sum_weights": 0.0,
                }
            projectedBins[key]["sum_weights"] += float(sumWeight)

        bins = []
        for index, key in enumerate(sorted(projectedBins)):
            projectedBin = projectedBins[key]
            bins.append(
                {
                    "index": index,
                    "x_min": projectedBin["x_min"],
                    "x_max": projectedBin["x_max"],
                    "y_min": projectedBin["y_min"],
                    "y_max": projectedBin["y_max"],
                    "measure": projectedBin["measure"],
                    "sum_weights": projectedBin["sum_weights"],
                }
            )

        sumWeights = np.array(
            [binDefinition["sum_weights"] for binDefinition in bins], dtype=np.float64
        )
        binMeasures = np.array(
            [binDefinition["measure"] for binDefinition in bins], dtype=np.float64
        )
        values = np.divide(
            sumWeights,
            binMeasures,
            out=np.full_like(sumWeights, np.nan),
            where=binMeasures != 0,
        ) if divideByBinVolume else np.array(sumWeights, copy=True)

        return {
            "variable_names": variableNames,
            "bins": bins,
            "sum_weights": sumWeights,
            "bin_measures": binMeasures,
            "bin_volumes": binMeasures,
            "values": np.array(values, copy=True),
            "values_label": "sumWeights / binMeasure"
            if divideByBinVolume
            else "sumWeights",
            "x_edges": np.unique(
                [
                    edge
                    for binDefinition in bins
                    for edge in (
                        binDefinition["x_min"],
                        binDefinition["x_max"],
                    )
                ]
            ),
            "y_edges": np.unique(
                [
                    edge
                    for binDefinition in bins
                    for edge in (
                        binDefinition["y_min"],
                        binDefinition["y_max"],
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
