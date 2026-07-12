"""Helpers for restoring state from GUNDAM ROOT output files."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RootHistogramState:
    """Histogram bin state read from a GUNDAM ROOT output file."""

    sumWeights: np.ndarray
    sqrtSumSqWeights: np.ndarray


@dataclass(frozen=True, slots=True)
class GundamRootStateReader:
    """Read state payloads stored in a GUNDAM ROOT output file."""

    path: str | Path

    POST_FIT_PARAMETER_STATE_PATH = (
        "FitterEngine/postFit/Migrad/parameterStateAfterMinimize_TNamed"
    )
    DATA_HISTOGRAM_PATH_TEMPLATE = "FitterEngine/preFit/data/{sampleName}/histogram"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser())

    def readPostFitParameterStateString(self) -> str:
        """Return the serialized post-fit parameter injector config."""
        return self._readTNamedTitle(self.POST_FIT_PARAMETER_STATE_PATH)

    def buildPostFitParameterStateConfig(self, gundam: Any) -> Any:
        """Return a live ``ConfigBuilder`` holding the post-fit state config."""
        return self._jsonStringToConfigBuilder(
            gundam,
            self.readPostFitParameterStateString(),
        )

    def readDataHistogram(self, sampleName: str) -> RootHistogramState:
        """Return the saved pre-fit data histogram for one sample."""
        objectPath = self.DATA_HISTOGRAM_PATH_TEMPLATE.format(sampleName=sampleName)
        return self._readHistogram(objectPath)

    def _readTNamedTitle(self, objectPath: str) -> str:
        import uproot

        try:
            with uproot.open(self.path) as rootFile:
                try:
                    named = rootFile[objectPath]
                except KeyError as error:
                    raise KeyError(
                        f"Could not find '{objectPath}' in GUNDAM ROOT output file: {self.path}"
                    ) from error
                try:
                    return str(named.member("fTitle"))
                except (AttributeError, KeyError) as error:
                    raise TypeError(
                        f"Object '{objectPath}' in {self.path} does not expose TNamed.fTitle"
                    ) from error
        except OSError as error:
            raise FileNotFoundError(
                f"Could not open GUNDAM ROOT output file: {self.path}"
            ) from error

    def _readHistogram(self, objectPath: str) -> RootHistogramState:
        import uproot

        try:
            with uproot.open(self.path) as rootFile:
                try:
                    histogram = rootFile[objectPath]
                except KeyError as error:
                    raise KeyError(
                        f"Could not find '{objectPath}' in GUNDAM ROOT output file: {self.path}"
                    ) from error
                try:
                    sumWeights = np.asarray(histogram.values(flow=False), dtype=np.float64)
                    sqrtSumSqWeights = np.asarray(histogram.errors(flow=False), dtype=np.float64)
                except AttributeError as error:
                    raise TypeError(
                        f"Object '{objectPath}' in {self.path} does not expose TH1 values"
                    ) from error
                return RootHistogramState(
                    sumWeights=sumWeights,
                    sqrtSumSqWeights=sqrtSumSqWeights,
                )
        except OSError as error:
            raise FileNotFoundError(
                f"Could not open GUNDAM ROOT output file: {self.path}"
            ) from error

    @staticmethod
    def _jsonStringToConfigBuilder(gundam: Any, jsonString: str) -> Any:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=True,
        ) as jsonFile:
            jsonFile.write(jsonString)
            jsonFile.flush()
            return gundam.ConfigUtils.ConfigBuilder(str(jsonFile.name))
