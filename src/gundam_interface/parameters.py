from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class GundamParameter:
    """Light Python-side view over a GUNDAM Parameter handle."""

    _handle: Any

    def getName(self) -> str:
        return str(self._handle.getName())

    def getFullTitle(self) -> str:
        return str(self._handle.getFullTitle())

    def isEnabled(self) -> bool:
        return bool(self._handle.isEnabled())

    def getStepSize(self) -> float:
        return float(self._handle.getStepSize())

    def getPrior(self) -> float:
        return float(self._handle.getPriorValue())

    def getThrow(self) -> float:
        return float(self._handle.getThrowValue())

    def getValue(self) -> float:
        return float(self._handle.getParameterValue())

    def setValue(self, value: float) -> None:
        self._handle.setParameterValue(float(value), True)


@dataclass(slots=True)
class GundamParameterSet:
    """Light Python-side view over a GUNDAM ParameterSet handle."""

    _handle: Any

    @property
    def isEnableEigenDecomp(self) -> bool:
        return bool(self._handle.isEnableEigenDecomp())

    @property
    def parameters(self) -> list[GundamParameter]:
        parameters: list[GundamParameter] = []
        for parameter in self._handle.getParameterList():
            parameters.append(GundamParameter(_handle=parameter))
        return parameters

    @property
    def eigenParameters(self) -> list[GundamParameter]:
        parameters: list[GundamParameter] = []
        for parameter in self._handle.getEigenParameterList():
            parameters.append(GundamParameter(_handle=parameter))
        return parameters

    @property
    def priorCovarianceMatrix(self) -> Any:
        return self._handle.getPriorCovarianceMatrix()

    @property
    def priorFullCovarianceMatrix(self) -> Any:
        return self._handle.getPriorFullCovarianceMatrix()

    def propagateOriginalToEigen(self) -> None:
        self._handle.propagateOriginalToEigen()

    def propagateEigenToOriginal(self) -> None:
        self._handle.propagateEigenToOriginal()


def wrapParameterSetList(parameterSets: Any) -> list[GundamParameterSet]:
    return [GundamParameterSet(_handle=parameterSet) for parameterSet in parameterSets]


def collectActiveParameters(
    parametersManager: Any,
) -> list[GundamParameter]:
    parameters: list[GundamParameter] = []
    for parameterSet in parametersManager.getParameterSetsList():
        for parameter in parameterSet.getParameterList():
            if not parameter.isEnabled():
                continue

            stepSize = float(parameter.getStepSize())
            if not np.isfinite(stepSize) or stepSize <= 0:
                raise ValueError(
                    f"Invalid step size for {parameter.getFullTitle()}: {stepSize}"
                )

            parameters.append(
                GundamParameter(_handle=parameter)
            )
    return parameters


def parameterPriors(parameters: list[GundamParameter]) -> np.ndarray:
    return np.array([parameter.getPrior() for parameter in parameters], dtype=np.float64)


def parameterSteps(parameters: list[GundamParameter]) -> np.ndarray:
    return np.array([parameter.getStepSize() for parameter in parameters], dtype=np.float64)


def parameterThrowValues(
    parameters: list[GundamParameter],
    *,
    includeThrowValues: bool = False,
) -> np.ndarray | None:
    if not includeThrowValues:
        return None
    return np.array([parameter.getThrow() for parameter in parameters], dtype=np.float64)
