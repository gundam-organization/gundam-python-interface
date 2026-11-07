from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .parameters import ParameterView


@dataclass(slots=True)
class GundamMinimizer:
    """Light Python-side interface around the GUNDAM minimizer."""

    _handle: Any

    def getFitParameters(self) -> list[ParameterView]:
        return [ParameterView(fitPar) for fitPar in self._handle.getMinimizerFitParameterPtr()]

    def minimize(self) -> None:
        self._handle.minimize()

    def throwPostfitParameters(self) -> None:
        self._handle.throwPostfitParameters()

    def getMinimizerStatus(self) -> int:
        return self._handle.getMinimizerStatus()

    def isRootMinimizer(self) -> bool:
        """
        Whether the C++ handle is a RootMinimizer or one of
        its subclasses.
        """
        return self._handle.isRootMinimizer()

    def _requireRootMinimizer(self) -> None:
        if not self.isRootMinimizer():
            raise TypeError(
                "This method requires a RootMinimizer; "
                f"got {type(self._handle).__name__}."
            )

    def updateFitParameterSettings(self) -> None:
        """Sync ROOT variables with current parameter values, steps, limits and fixed states."""
        self._requireRootMinimizer()
        self._handle.updateFitParameterSettings()

    def getFunctionPrecision(self) -> float:
        self._requireRootMinimizer()
        return self._handle.getFunctionPrecision()

    def hasConverged(self) -> bool:
        self._requireRootMinimizer()
        return self._handle.hasConverged()

    def isMinimizeDone(self) -> bool:
        self._requireRootMinimizer()
        return self._handle.isMinimizeDone()
