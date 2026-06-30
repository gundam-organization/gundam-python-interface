from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class GundamParameter:
    """Small Python-side descriptor for an enabled GUNDAM parameter."""

    index: int
    parameterSetIndex: int
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


def getParameterThrowValue(parameter: Any) -> float:
    return float(parameter.getThrowValue())


def collectActiveParameters(
    parametersManager: Any,
    *,
    includeThrowValues: bool = False,
) -> list[GundamParameter]:
    parameters: list[GundamParameter] = []
    for parameterSetIndex, parameterSet in enumerate(parametersManager.getParameterSetsList()):
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
                    parameterSetIndex=parameterSetIndex,
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


def normalizedToPhysical(
    normalizedValues: np.ndarray,
    priors: np.ndarray,
    stepSizes: np.ndarray,
) -> np.ndarray:
    return priors + stepSizes * np.asarray(normalizedValues, dtype=np.float64)


def physicalToNormalized(
    physicalValues: np.ndarray,
    priors: np.ndarray,
    stepSizes: np.ndarray,
) -> np.ndarray:
    return (np.asarray(physicalValues, dtype=np.float64) - priors) / stepSizes
