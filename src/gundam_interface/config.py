from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GundamRuntime:
    """Runtime context needed to construct the GUNDAM Python interface."""

    workDir: str | Path
    nCpuThreads: int = 1
    configPath: str | Path | None = None
    overrideList: list[str | Path] = field(default_factory=list)
    configJsonString: str | None = None
    forceAsimov: bool | None = None
    dataType: str | None = None
    randomSeed: int | None = None

    def __post_init__(self) -> None:
        self.workDir = Path(self.workDir).expanduser()
        if self.configPath is not None:
            self.configPath = Path(self.configPath).expanduser()
        self.overrideList = [Path(path).expanduser() for path in self.overrideList]
        if self.configJsonString is not None:
            self.configJsonString = self.configJsonString.strip()

        if self.nCpuThreads < 1:
            raise ValueError("nCpuThreads must be >= 1")
        if self.randomSeed is not None:
            self.randomSeed = int(self.randomSeed)
            if self.randomSeed < 0:
                raise ValueError("randomSeed must be >= 0")
        if self.configPath is None and self.configJsonString is None:
            raise ValueError("Either configPath or configJsonString must be provided")
        self.dataType = self._canonicalDataType(self.dataType, self.forceAsimov)

    @classmethod
    def fromDict(cls, data: dict[str, Any]) -> "GundamRuntime":
        return cls(
            workDir=data["workDir"],
            nCpuThreads=int(data.get("nCpuThreads", 1)),
            configPath=data.get("configPath"),
            overrideList=list(data.get("overrideList", [])),
            configJsonString=data.get("configJsonString"),
            forceAsimov=data.get("forceAsimov", data.get("useAsimov")),
            dataType=data.get("dataType"),
            randomSeed=data.get("randomSeed", data.get("seed")),
        )

    @classmethod
    def fromJsonFile(cls, path: str | Path) -> "GundamRuntime":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls.fromDict(json.load(file))

    def toDict(self, includeConfigJsonString: bool = True) -> dict[str, Any]:
        data = {
            "nCpuThreads": self.nCpuThreads,
            "workDir": str(self.workDir),
            "dataType": self.dataType,
        }
        if self.randomSeed is not None:
            data["randomSeed"] = self.randomSeed
        if self.configJsonString is not None:
            if includeConfigJsonString:
                data["configJsonString"] = self.configJsonString
        else:
            data["configPath"] = str(self.configPath)
            data["overrideList"] = [str(path) for path in self.overrideList]
        return data

    def toJsonFile(self, path: str | Path) -> None:
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
        if self.configPath is None:
            raise ValueError("No configPath is defined for this GundamRuntime")
        if self.configPath.is_absolute():
            return self.configPath
        return self.workDir / self.configPath

    @property
    def absoluteOverridePaths(self) -> list[Path]:
        overridePaths = []
        for overridePath in self.overrideList:
            if overridePath.is_absolute():
                overridePaths.append(overridePath)
            else:
                overridePaths.append(self.workDir / overridePath)
        return overridePaths

    @property
    def defaultInitializeLogPath(self) -> Path:
        return self.workDir / "gundam_initialize.log"

    @property
    def defaultEvaluateLogPath(self) -> Path:
        return self.workDir / "gundam_evaluate.log"

    def validatePaths(self) -> None:
        """Fail early on missing user-provided paths."""
        if not self.workDir.exists():
            raise FileNotFoundError(f"GUNDAM workDir does not exist: {self.workDir}")
        if self.configJsonString is not None:
            return
        if not self.absoluteConfigPath.exists():
            raise FileNotFoundError(f"GUNDAM config file does not exist: {self.absoluteConfigPath}")
        for overridePath in self.absoluteOverridePaths:
            if not overridePath.exists():
                raise FileNotFoundError(f"GUNDAM override file does not exist: {overridePath}")
