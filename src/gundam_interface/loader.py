from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def setupPythonPath(pythonPath: str | os.PathLike[str]) -> Path:
    """Add a GUNDAM Python binding directory to Python import paths."""
    path = Path(pythonPath).expanduser()
    pathString = str(path)

    if pathString not in sys.path:
        sys.path.insert(0, pathString)

    existingPythonPath = os.environ.get("PYTHONPATH", "")
    pythonPathParts = [part for part in existingPythonPath.split(os.pathsep) if part]
    if pathString not in pythonPathParts:
        os.environ["PYTHONPATH"] = os.pathsep.join([pathString, *pythonPathParts])

    return path


def importGundam(pythonPath: str | os.PathLike[str] | None = None) -> ModuleType:
    """Import and return the GUNDAM Python bindings."""
    if pythonPath is not None:
        setupPythonPath(pythonPath)
    return importlib.import_module("GUNDAM")
