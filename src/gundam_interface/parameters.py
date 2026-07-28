from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class GundamParameter:
    """Light Python-side view over a GUNDAM Parameter handle."""

    _handle: Any

    @property
    def name(self) -> str:
        return str(self._handle.getName())

    @property
    def getFullTitle(self):
        return str(self._handle.getFullTitle())

    @property
    def isEnabled(self) -> bool:
        return bool(self._handle.isEnabled())

    @property
    def stepSize(self) -> float:
        return float(self._handle.getStepSize())

    @property
    def prior(self) -> float:
        return float(self._handle.getPriorValue())

    @property
    def throwValue(self) -> float:
        return float(self._handle.getThrowValue())

    @property
    def value(self) -> float:
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


def getParameterThrowValue(parameter: Any) -> float:
    return float(parameter.getThrowValue())


def wrapParameterSetList(parameterSets: Any) -> list[GundamParameterSet]:
    return [GundamParameterSet(_handle=parameterSet) for parameterSet in parameterSets]


def collectActiveParameters(
    parametersManager: Any,
    *,
    includeThrowValues: bool = False,
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
    return np.array([parameter.prior for parameter in parameters], dtype=np.float64)


def parameterSteps(parameters: list[GundamParameter]) -> np.ndarray:
    return np.array([parameter.stepSize for parameter in parameters], dtype=np.float64)


def parameterThrowValues(parameters: list[GundamParameter]) -> np.ndarray | None:
    try:
        values = [parameter.throwValue for parameter in parameters]
    except Exception:
        return None
    return np.array(values, dtype=np.float64)
