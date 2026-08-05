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
