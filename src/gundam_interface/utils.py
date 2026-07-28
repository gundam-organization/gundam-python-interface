from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterator

import numpy as np


@dataclass(slots=True)
class GundamCovarianceMatrix:
    """Light Python-side view over a GUNDAM TMatrixDSym handle."""

    _handle: Any

    def getNrows(self) -> int:
        return int(self._handle.GetNrows())

    def getNcols(self) -> int:
        return int(self._handle.GetNcols())

    def getValue(self, row: int, col: int) -> float:
        return float(self._handle[row, col])

    def setValue(self, row: int, col: int, value: float) -> None:
        self._handle[row, col] = float(value)

    def toNumpyArray(self) -> np.ndarray:
        out = np.empty((self.getNrows(), self.getNcols()), dtype=np.float64)
        for row in range(out.shape[0]):
            for col in range(out.shape[1]):
                out[row, col] = self.getValue(row, col)
        return out


@contextmanager
def preservedWorkingDirectory() -> Iterator[None]:
    originalWorkingDirectory = Path.cwd()
    try:
        yield
    finally:
        os.chdir(originalWorkingDirectory)


@contextmanager
def temporaryWorkingDirectory(path: str | os.PathLike[str]) -> Iterator[None]:
    originalWorkingDirectory = Path.cwd()
    os.chdir(Path(path).expanduser().resolve())
    try:
        yield
    finally:
        os.chdir(originalWorkingDirectory)
