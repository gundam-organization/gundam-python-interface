"""Utilities for importing the GUNDAM Python bindings."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(slots=True)
class GundamLoader:
    """Import helper for the GUNDAM Python bindings.

    ``GundamLoader`` owns the information required to make ``import GUNDAM``
    work in the current Python process. In typical local installations,
    ``gundamLibPath`` points to the ``lib`` directory of the GUNDAM
    installation, the directory that contains the GUNDAM Python module.

    The loader updates both ``sys.path`` and the ``PYTHONPATH`` environment
    variable before importing the configured module. This mirrors the manual
    setup usually needed in notebooks and scripts while keeping that setup out
    of ``GundamRuntime``.

    Parameters
    ----------
    gundamLibPath:
        Path to the GUNDAM installation ``lib`` directory. If omitted, the
        loader assumes the GUNDAM Python bindings are already importable.
    moduleName:
        Python module name to import. Defaults to ``"GUNDAM"``.
    """

    gundamLibPath: str | os.PathLike[str] | None = None
    moduleName: str = "GUNDAM"

    def __post_init__(self) -> None:
        if self.gundamLibPath is not None:
            self.gundamLibPath = Path(self.gundamLibPath).expanduser()

    def setupPythonPath(self) -> Path | None:
        """Add ``gundamLibPath`` to Python import paths.

        Returns
        -------
        Path | None
            The resolved GUNDAM lib path when one was configured, otherwise
            ``None``.
        """
        if self.gundamLibPath is None:
            return None
        path = Path(self.gundamLibPath).expanduser()
        pathString = str(path)

        if pathString not in sys.path:
            sys.path.insert(0, pathString)

        existingPythonPath = os.environ.get("PYTHONPATH", "")
        pythonPathParts = [part for part in existingPythonPath.split(os.pathsep) if part]
        if pathString not in pythonPathParts:
            os.environ["PYTHONPATH"] = os.pathsep.join([pathString, *pythonPathParts])

        return path

    def importGundam(self) -> ModuleType:
        """Import and return the configured GUNDAM module."""
        self.setupPythonPath()
        return importlib.import_module(self.moduleName)

    def toDict(self) -> dict[str, str]:
        """Return a JSON-compatible loader description."""
        data = {"moduleName": self.moduleName}
        if self.gundamLibPath is not None:
            data["gundamLibPath"] = str(self.gundamLibPath)
        return data

    @classmethod
    def fromDict(cls, data: dict[str, str] | None) -> "GundamLoader":
        """Create a loader from serialized metadata.

        ``gundamLibPath`` is the preferred key. Legacy metadata using
        ``pythonPath`` is still accepted.
        """
        if data is None:
            return cls()
        return cls(
            gundamLibPath=data.get("gundamLibPath", data.get("pythonPath")),
            moduleName=data.get("moduleName", "GUNDAM"),
        )
