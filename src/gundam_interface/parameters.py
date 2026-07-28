from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import GundamCovarianceMatrix


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

    def isEnableEigenDecomp(self) -> bool:
        return bool(self._handle.isEnableEigenDecomp())

    def getParameterList(self) -> list[GundamParameter]:
        out: list[GundamParameter] = []
        for parameter in self._handle.getParameterList():
            out.append(GundamParameter(_handle=parameter))
        return out

    def getEigenParameterList(self) -> list[GundamParameter]:
        if not self._handle.isEnableEigenDecomp():
            return []
        out: list[GundamParameter] = []
        for parameter in self._handle.getEigenParameterList():
            out.append(GundamParameter(_handle=parameter))
        return out

    def getPriorCovarianceMatrix(self) -> Any:
        return GundamCovarianceMatrix(self._handle.getPriorCovarianceMatrix())


@dataclass(slots=True)
class GundamParametersManager:
    """Light Python-side view over a GUNDAM ParametersManager handle."""

    _handle: Any

    def throwParameters(self) -> None:
        self._handle.throwParameters()

    def exportParametersStateJson(self) -> str:
        return json.loads(self._handle.exportParameterInjectorConfig().toString())

    def injectParametersState(self, jsonString_: str) -> None:
        self._handle.injectParameterValues(jsonString_)

    def getParameterSetList(self) -> list[GundamParameterSet]:
        return [
            GundamParameterSet(_handle=parameterSet)
            for parameterSet in self._handle.getParameterSetsList()
        ]

    def getActiveParameterList(self) -> list[GundamParameter]:
        out: list[GundamParameter] = []
        for parameterSet in self.getParameterSetList():
            for parameter in parameterSet.getParameterList():
                if parameter.isEnabled():
                    out.append(parameter)
        return out

    def getParameterValues(self) -> np.ndarray:
        return np.array(
            [parameter.getValue() for parameter in self.getActiveParameterList()],
            dtype=np.float64,
        )

    def setParameterValues(self, values: np.ndarray) -> None:
        parameters = self.getActiveParameterList()
        values = np.asarray(values, dtype=np.float64)
        expectedShape = (len(parameters),)
        if values.shape != expectedShape:
            raise ValueError(f"Expected parameter shape {expectedShape}, got {values.shape}")
        for parameter, value in zip(parameters, values):
            parameter.setValue(float(value))

    def resetToPrior(self) -> None:
        for parameter in self.getActiveParameterList():
            parameter.setValue(parameter.getPrior())
