from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class GundamParameter:
    """Small Python-side descriptor for an enabled GUNDAM parameter."""

    index: int
    parameterIndex: int
    name: str
    prior: float
    stepSize: float
    throwValue: float | None
    handle: Any

    @property
    def value(self) -> float:
        return float(self.handle.getParameterValue())

    def setValue(self, value: float) -> None:
        self.handle.setParameterValue(float(value), True)

    def resetToPrior(self) -> None:
        self.setValue(self.prior)


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
        for parameterIndex, parameter in enumerate(self._handle.getParameterList()):
            parameters.append(
                GundamParameter(
                    index=parameterIndex,
                    parameterIndex=parameterIndex,
                    name=parameter.getFullTitle(),
                    prior=float(parameter.getPriorValue()),
                    stepSize=float(parameter.getStepSize()),
                    throwValue=None,
                    handle=parameter,
                )
            )
        return parameters

    @property
    def eigenParameters(self) -> list[GundamParameter]:
        parameters: list[GundamParameter] = []
        for parameterIndex, parameter in enumerate(self._handle.getEigenParameterList()):
            parameters.append(
                GundamParameter(
                    index=parameterIndex,
                    parameterIndex=parameterIndex,
                    name=parameter.getFullTitle(),
                    prior=float(parameter.getPriorValue()),
                    stepSize=float(parameter.getStepSize()),
                    throwValue=None,
                    handle=parameter,
                )
            )
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
        for parameterIndex, parameter in enumerate(parameterSet.getParameterList()):
            if not parameter.isEnabled():
                continue

            stepSize = float(parameter.getStepSize())
            if not np.isfinite(stepSize) or stepSize <= 0:
                raise ValueError(
                    f"Invalid step size for {parameter.getFullTitle()}: {stepSize}"
                )

            parameters.append(
                GundamParameter(
                    index=len(parameters),
                    parameterIndex=parameterIndex,
                    name=parameter.getFullTitle(),
                    prior=float(parameter.getPriorValue()),
                    stepSize=stepSize,
                    throwValue=(
                        getParameterThrowValue(parameter) if includeThrowValues else None
                    ),
                    handle=parameter,
                )
            )
    return parameters


def parameterPriors(parameters: list[GundamParameter]) -> np.ndarray:
    return np.array([parameter.prior for parameter in parameters], dtype=np.float64)


def parameterSteps(parameters: list[GundamParameter]) -> np.ndarray:
    return np.array([parameter.stepSize for parameter in parameters], dtype=np.float64)


def parameterThrowValues(parameters: list[GundamParameter]) -> np.ndarray | None:
    values = [parameter.throwValue for parameter in parameters]
    if any(value is None for value in values):
        return None
    return np.array(values, dtype=np.float64)
