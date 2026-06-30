from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(slots=True)
class GundamLoader:
    """Loader responsible for making the GUNDAM Python bindings importable."""

    pythonPath: str | os.PathLike[str] | None = None
    moduleName: str = "GUNDAM"

    def __post_init__(self) -> None:
        if self.pythonPath is not None:
            self.pythonPath = Path(self.pythonPath).expanduser()

    def setupPythonPath(self) -> Path | None:
        if self.pythonPath is None:
            return None
        path = Path(self.pythonPath).expanduser()
        pathString = str(path)

        if pathString not in sys.path:
            sys.path.insert(0, pathString)

        existingPythonPath = os.environ.get("PYTHONPATH", "")
        pythonPathParts = [part for part in existingPythonPath.split(os.pathsep) if part]
        if pathString not in pythonPathParts:
            os.environ["PYTHONPATH"] = os.pathsep.join([pathString, *pythonPathParts])

        return path

    def importGundam(self) -> ModuleType:
        self.setupPythonPath()
        return importlib.import_module(self.moduleName)

    def toDict(self) -> dict[str, str]:
        data = {"moduleName": self.moduleName}
        if self.pythonPath is not None:
            data["pythonPath"] = str(self.pythonPath)
        return data

    @classmethod
    def fromDict(cls, data: dict[str, str] | None) -> "GundamLoader":
        if data is None:
            return cls()
        return cls(
            pythonPath=data.get("pythonPath"),
            moduleName=data.get("moduleName", "GUNDAM"),
        )
