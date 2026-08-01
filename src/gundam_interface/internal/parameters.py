from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import CovarianceMatrixView


@dataclass(slots=True)
class ParameterView:
    """Light Python-side view over a GUNDAM Parameter handle."""

    handle: Any

    def getName(self) -> str:
        return str(self.handle.getName())

    def getFullTitle(self) -> str:
        return str(self.handle.getFullTitle())

    def isEnabled(self) -> bool:
        return bool(self.handle.isEnabled())

    def getStepSize(self) -> float:
        return float(self.handle.getStepSize())

    def getPrior(self) -> float:
        return float(self.handle.getPriorValue())

    def getThrow(self) -> float:
        return float(self.handle.getThrowValue())

    def getValue(self) -> float:
        return float(self.handle.getParameterValue())

    def getSummary(self) -> str:
        return str(self.handle.getSummary())

    def setValue(self, value: float) -> None:
        self.handle.setParameterValue(float(value), True)


@dataclass(slots=True)
class ParameterSetView:
    """Light Python-side view over a GUNDAM ParameterSet handle."""

    handle: Any

    def isEnableEigenDecomp(self) -> bool:
        return bool(self.handle.isEnableEigenDecomp())

    def getParameterList(self) -> list[ParameterView]:
        out: list[ParameterView] = []
        for parameter in self.handle.getParameterList():
            out.append(ParameterView(handle=parameter))
        return out

    def getEigenParameterList(self) -> list[ParameterView]:
        if not self.handle.isEnableEigenDecomp():
            return []
        out: list[ParameterView] = []
        for parameter in self.handle.getEigenParameterList():
            out.append(ParameterView(handle=parameter))
        return out

    def getPriorCovarianceMatrix(self) -> Any:
        return CovarianceMatrixView(self.handle.getPriorCovarianceMatrix())


@dataclass(slots=True)
class ParametersManagerView:
    """Light Python-side view over a GUNDAM ParametersManager handle."""

    handle: Any

    def throwParameters(self) -> None:
        self.handle.throwParameters()

    def exportParametersStateJson(self) -> str:
        return json.loads(self.handle.exportParameterInjectorConfig().toString())

    def injectParametersState(self, jsonString_: str) -> None:
        self.handle.injectParameterValues(jsonString_)

    def getParameterSetList(self) -> list[ParameterSetView]:
        return [
            ParameterSetView(handle=parameterSet)
            for parameterSet in self.handle.getParameterSetsList()
        ]

    def getActiveParameterList(self) -> list[ParameterView]:
        out: list[ParameterView] = []
        for parameterSet in self.getParameterSetList():
            parList = parameterSet.getParameterList() if not parameterSet.isEnableEigenDecomp() else parameterSet.getEigenParameterList()
            for parameter in parList:
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
