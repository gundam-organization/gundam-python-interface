"""Runtime configuration for a GUNDAM engine instance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .loader import GundamLoader
from .logging import GundamLogRedirector


@dataclass(slots=True)
class GundamRuntime:
    """Configuration needed to load and configure a GUNDAM engine.

    ``GundamRuntime`` describes how a ``GundamInterface`` should construct a
    concrete GUNDAM engine instance. It keeps together the loader used to import
    the GUNDAM Python bindings, the runtime working directory, the GUNDAM
    configuration source, override files, data mode, thread count, and optional
    random seed.

    The runtime accepts a configuration file path relative to ``workDir``, an
    already-built JSON configuration string, or a GUNDAM ROOT output file.
    ``configPath`` takes priority over ``outputRootPath`` as the configuration
    source when both are provided. Override paths are resolved relative to
    ``workDir`` unless they are absolute.

    Parameters
    ----------
    loader:
        Loader used to import the GUNDAM Python bindings.
    workDir:
        GUNDAM runtime working directory. Relative config and override paths are
        resolved from this directory.
    nCpuThreads:
        Number of CPU threads to request from GUNDAM. Defaults to 1.
    configPath:
        Path to the base GUNDAM config file, relative to ``workDir`` or
        absolute. Mutually exclusive with ``configJsonString``. Takes priority
        over ``outputRootPath`` as the configuration source.
    overrideList:
        Sequence of override config files applied after the base config source.
    configJsonString:
        Serialized GUNDAM JSON config. Mutually exclusive with ``configPath``.
    outputRootPath:
        Path to a GUNDAM ROOT output file, relative to ``workDir`` or absolute.
        Used as the configuration source when neither ``configPath`` nor
        ``configJsonString`` is set. It is also used as the pre-fit data
        histogram source when ``loadDataHistograms`` is true, and as the
        post-fit state source when ``loadPostFitState`` is true.
    loadDataHistograms:
        Load saved pre-fit data histograms from ``outputRootPath`` after engine
        initialization. Defaults to ``True`` so a ROOT state load preserves the
        original data, including toys whose seed is not available. Set to
        ``False`` to load only the stored config and let GUNDAM build data from
        ``dataType`` and ``randomSeed``.
    loadPostFitState:
        Load and inject the post-fit parameter state from ``outputRootPath``
        after engine initialization. Defaults to ``False``.
    forceAsimov:
        Backward-compatible way to choose Asimov or real data when ``dataType``
        is omitted.
    dataType:
        GUNDAM likelihood data type: ``"Asimov"``, ``"Toy"``, or
        ``"RealData"``. Defaults to ``"Asimov"``.
    randomSeed:
        Optional non-negative seed applied to the GUNDAM engine.
    """

    loader: GundamLoader
    workDir: str | Path
    nCpuThreads: int = 1
    configPath: str | Path | None = None
    overrideList: list[str | Path] = field(default_factory=list)
    configJsonString: str | None = None
    outputRootPath: str | Path | None = None
    loadDataHistograms: bool = True
    loadPostFitState: bool = False
    forceAsimov: bool | None = None
    dataType: str | None = None
    randomSeed: int | None = None
    logRedirector: GundamLogRedirector = field(
        init=False,
        default_factory=GundamLogRedirector,
    )

    def __post_init__(self) -> None:
        self.workDir = Path(self.workDir).expanduser()
        if isinstance(self.loader, dict):
            self.loader = GundamLoader.fromDict(self.loader)
        if self.configPath is not None:
            self.configPath = Path(self.configPath).expanduser()
        if self.outputRootPath is not None:
            self.outputRootPath = Path(self.outputRootPath).expanduser()
        self.overrideList = [Path(path).expanduser() for path in self.overrideList]
        if self.configJsonString is not None:
            self.configJsonString = self.configJsonString.strip()
        self.loadDataHistograms = bool(self.loadDataHistograms)
        self.loadPostFitState = bool(self.loadPostFitState)

        if self.nCpuThreads < 1:
            raise ValueError("nCpuThreads must be >= 1")
        if self.randomSeed is not None:
            self.randomSeed = int(self.randomSeed)
            if self.randomSeed < 0:
                raise ValueError("randomSeed must be >= 0")
        if self.configPath is not None and self.configJsonString is not None:
            raise ValueError("configPath and configJsonString are mutually exclusive")
        if (
            self.configPath is None
            and self.configJsonString is None
            and self.outputRootPath is None
        ):
            raise ValueError(
                "Either configPath, configJsonString, or outputRootPath must be provided"
            )
        if self.loadPostFitState and self.outputRootPath is None:
            raise ValueError("outputRootPath must be provided when loadPostFitState is True")
        if (
            self.outputRootPath is not None
            and not self.loadDataHistograms
            and self.dataType is None
            and self.forceAsimov is None
        ):
            raise ValueError(
                "dataType must be provided when loadDataHistograms is False"
            )
        self.dataType = self._canonicalDataType(self.dataType, self.forceAsimov)
        if (
            self.outputRootPath is not None
            and not self.loadDataHistograms
            and self.dataType == "Toy"
            and self.randomSeed is None
        ):
            raise ValueError(
                "randomSeed must be provided for Toy data when loadDataHistograms is False"
            )

    @classmethod
    def fromDict(cls, data: dict[str, Any]) -> "GundamRuntime":
        """Create a runtime from a JSON-compatible dictionary.

        The preferred loader schema is ``{"loader": {"gundamLibPath": ...}}``.
        Top-level ``gundamLibPath`` and legacy top-level ``pythonPath`` are also
        accepted for compatibility with older metadata.
        """
        return cls(
            workDir=data["workDir"],
            nCpuThreads=int(data.get("nCpuThreads", 1)),
            configPath=data.get("configPath"),
            overrideList=list(data.get("overrideList", [])),
            configJsonString=data.get("configJsonString"),
            outputRootPath=data.get("outputRootPath"),
            loadDataHistograms=bool(data.get("loadDataHistograms", True)),
            loadPostFitState=bool(data.get("loadPostFitState", False)),
            forceAsimov=data.get("forceAsimov", data.get("useAsimov")),
            dataType=data.get("dataType"),
            randomSeed=data.get("randomSeed", data.get("seed")),
            loader=GundamLoader.fromDict(
                data.get("loader")
                or (
                    {"gundamLibPath": data["gundamLibPath"]}
                    if data.get("gundamLibPath") is not None
                    else {"pythonPath": data["pythonPath"]}
                    if data.get("pythonPath") is not None
                    else None
                )
            ),
        )

    @classmethod
    def fromJsonFile(cls, path: str | Path) -> "GundamRuntime":
        """Load a runtime definition from a JSON file."""
        with Path(path).open("r", encoding="utf-8") as file:
            return cls.fromDict(json.load(file))

    def toDict(self, includeConfigJsonString: bool = True) -> dict[str, Any]:
        """Return a JSON-compatible runtime description.

        Parameters
        ----------
        includeConfigJsonString:
            Include the full serialized config when this runtime was built from
            ``configJsonString``. Set this to ``False`` when writing lightweight
            metadata that should avoid embedding the full config payload.
        """
        data = {
            "nCpuThreads": self.nCpuThreads,
            "workDir": str(self.workDir),
            "dataType": self.dataType,
            "loader": self.loader.toDict(),
        }
        if self.randomSeed is not None:
            data["randomSeed"] = self.randomSeed
        if self.outputRootPath is not None:
            data["outputRootPath"] = str(self.outputRootPath)
        if not self.loadDataHistograms:
            data["loadDataHistograms"] = self.loadDataHistograms
        if self.loadPostFitState:
            data["loadPostFitState"] = self.loadPostFitState
        if self.configJsonString is not None:
            if includeConfigJsonString:
                data["configJsonString"] = self.configJsonString
        elif self.configPath is not None:
            data["configPath"] = str(self.configPath)
        data["overrideList"] = [str(path) for path in self.overrideList]
        return data

    def toJsonFile(self, path: str | Path) -> None:
        """Write this runtime description as formatted JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.toDict(), file, indent=2, sort_keys=True)
            file.write("\n")

    @staticmethod
    def _canonicalDataType(dataType: str | None, forceAsimov: bool | None) -> str:
        if dataType is None:
            if forceAsimov is None or forceAsimov:
                return "Asimov"
            return "RealData"

        normalized = dataType.replace("_", "").replace("-", "").lower()
        aliases = {
            "asimov": "Asimov",
            "toy": "Toy",
            "realdata": "RealData",
            "real": "RealData",
            "data": "RealData",
        }
        try:
            return aliases[normalized]
        except KeyError as error:
            raise ValueError(
                "dataType must be one of: Asimov, Toy, RealData"
            ) from error

    @property
    def absoluteConfigPath(self) -> Path:
        """Resolved base config path."""
        if self.configPath is None:
            raise ValueError("No configPath is defined for this GundamRuntime")
        if self.configPath.is_absolute():
            return self.configPath
        return self.workDir / self.configPath

    @property
    def absoluteOutputRootPath(self) -> Path:
        """Resolved GUNDAM ROOT output path."""
        if self.outputRootPath is None:
            raise ValueError("No outputRootPath is defined for this GundamRuntime")
        if self.outputRootPath.is_absolute():
            return self.outputRootPath
        return self.workDir / self.outputRootPath

    @property
    def absoluteOverridePaths(self) -> list[Path]:
        """Resolved override config paths."""
        overridePaths = []
        for overridePath in self.overrideList:
            if overridePath.is_absolute():
                overridePaths.append(overridePath)
            else:
                overridePaths.append(self.workDir / overridePath)
        return overridePaths

    @property
    def defaultInitializeLogPath(self) -> Path:
        """Default log path for GUNDAM initialization output."""
        return self.workDir / "gundam_initialize.log"

    @property
    def defaultEvaluateLogPath(self) -> Path:
        """Default log path for GUNDAM likelihood evaluation output."""
        return self.workDir / "gundam_evaluate.log"

    def validatePaths(self) -> None:
        """Fail early when user-provided filesystem paths do not exist."""
        if not self.workDir.exists():
            raise FileNotFoundError(f"GUNDAM workDir does not exist: {self.workDir}")
        if self.configPath is not None and not self.absoluteConfigPath.exists():
            raise FileNotFoundError(f"GUNDAM config file does not exist: {self.absoluteConfigPath}")
        if self.outputRootPath is not None and not self.absoluteOutputRootPath.exists():
            raise FileNotFoundError(
                f"GUNDAM output ROOT file does not exist: {self.absoluteOutputRootPath}"
            )
        for overridePath in self.absoluteOverridePaths:
            if not overridePath.exists():
                raise FileNotFoundError(f"GUNDAM override file does not exist: {overridePath}")
