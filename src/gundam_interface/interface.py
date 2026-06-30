from __future__ import annotations

import importlib
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import GundamContext
from .logging import maybeRedirectNativeOutput, temporaryRedirectNativeOutput
from .parameters import (
    GundamParameter,
    collectActiveParameters,
    normalizedToPhysical,
    parameterPriors,
    parameterSteps,
    parameterThrowValues,
    physicalToNormalized,
)


@dataclass(frozen=True, slots=True)
class PostfitThrowSamples:
    """GUNDAM post-fit throws with propagated likelihood evaluations."""

    physicalValues: np.ndarray
    normalizedValues: np.ndarray
    llh: np.ndarray


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


class GundamInterface:
    """Thin Python wrapper around the GUNDAM fitting interface."""

    def __init__(self, context: GundamContext):
        self.context = context
        self.gundam: Any | None = None
        self.configBuilder: Any | None = None
        self.configJsonString: str | None = None
        self.fitterEngineConfig: Any | None = None
        self.engine: Any | None = None
        self.parameters: list[GundamParameter] = []

    @property
    def isConfigured(self) -> bool:
        return self.engine is not None

    @property
    def isInitialized(self) -> bool:
        return bool(self.parameters)

    @property
    def priors(self) -> np.ndarray:
        return parameterPriors(self.parameters)

    @property
    def stepSizes(self) -> np.ndarray:
        return parameterSteps(self.parameters)

    @property
    def throwValues(self) -> np.ndarray | None:
        return parameterThrowValues(self.parameters)

    @property
    def parameterNames(self) -> list[str]:
        return [parameter.name for parameter in self.parameters]

    def setupPythonPath(self) -> None:
        pythonPath = str(self.context.pythonPath)
        if pythonPath not in sys.path:
            sys.path.insert(0, pythonPath)

        existingPythonPath = os.environ.get("PYTHONPATH", "")
        pythonPathParts = [part for part in existingPythonPath.split(os.pathsep) if part]
        if pythonPath not in pythonPathParts:
            os.environ["PYTHONPATH"] = os.pathsep.join([pythonPath, *pythonPathParts])

    def importGundam(self):
        self.setupPythonPath()
        self.gundam = importlib.import_module("GUNDAM")
        return self.gundam

    def configure(self, validatePaths: bool = True) -> None:
        with preservedWorkingDirectory():
            if validatePaths:
                self.context.validatePaths()

            gundam = self.importGundam()
            gundam.setLightOutputMode(False)
            gundam.setNumberOfThreads(self.context.nCpuThreads)
            workingDirectory = Path(self.context.workDir).expanduser().resolve()
            gundam.setRuntimeWorkingDirectory(str(workingDirectory))

            with temporaryWorkingDirectory(workingDirectory):
                configBuilder = self._buildConfigBuilder(gundam)
                configJsonString = configBuilder.toString()

                configReader = gundam.ConfigUtils.ConfigReader(configBuilder.getConfig())
                configReader.defineField(
                    gundam.ConfigUtils.ConfigReader.FieldDefinition("fitterEngineConfig")
                )
                fitterEngineConfig = configReader.fetchValueConfigReader("fitterEngineConfig")

            engine = gundam.FitterEngine()
            engine.setConfig(fitterEngineConfig)
            self._setEngineRandomSeed(engine, self.context.randomSeed)
            with temporaryWorkingDirectory(workingDirectory):
                engine.configure()

            self.configBuilder = configBuilder
            self.configJsonString = configJsonString
            self.fitterEngineConfig = fitterEngineConfig
            self.engine = engine

    def _buildConfigBuilder(self, gundam):
        if self.context.configJsonString is not None:
            return self._buildConfigBuilderFromJsonString(gundam, self.context.configJsonString)

        configPath = Path(self.context.absoluteConfigPath).expanduser().resolve()
        overridePaths = [
            Path(overridePath).expanduser().resolve()
            for overridePath in self.context.absoluteOverridePaths
        ]
        configBuilder = gundam.ConfigUtils.ConfigBuilder(str(configPath))
        for overridePath in overridePaths:
            configBuilder.override(str(overridePath))
        return configBuilder

    @staticmethod
    def _buildConfigBuilderFromJsonString(gundam, configJsonString: str):
        # The Python binding exposes ConfigBuilder(str), but that overload expects a file path.
        # Keep the public API string-based and isolate the temporary bridge here.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=True,
        ) as configFile:
            configFile.write(configJsonString)
            configFile.flush()
            return gundam.ConfigUtils.ConfigBuilder(str(configFile.name))

    def initialize(
        self,
        logPath: str | os.PathLike[str] | None = None,
    ) -> None:
        with preservedWorkingDirectory():
            self._requireConfigured()
            workingDirectory = Path(self.context.workDir).expanduser().resolve()

            if logPath is None:
                redirectContext = temporaryRedirectNativeOutput("gundam_initialize")
            else:
                logPath = Path(logPath).expanduser().resolve()
                redirectContext = maybeRedirectNativeOutput(logPath)

            with temporaryWorkingDirectory(workingDirectory):
                with redirectContext:
                    self._setLikelihoodDataType()
                    self.engine.initialize()

            self.refreshParameters()

    def refreshParameters(self) -> list[GundamParameter]:
        self._requireConfigured()
        parametersManager = (
            self.engine.getLikelihoodInterface()
            .getModelPropagator()
            .getParametersManager()
        )
        self.parameters = collectActiveParameters(
            parametersManager,
            includeThrowValues=self.context.dataType == "Toy",
        )
        return self.parameters

    def getParameterValues(self) -> np.ndarray:
        self._requireParameters()
        return np.array([parameter.value for parameter in self.parameters], dtype=np.float64)

    def setParameterValues(self, values: np.ndarray) -> None:
        self._requireParameters()
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self.priors.shape:
            raise ValueError(f"Expected parameter shape {self.priors.shape}, got {values.shape}")
        for parameter, value in zip(self.parameters, values):
            parameter.setValue(float(value))

    def resetToPrior(self) -> None:
        self._requireParameters()
        for parameter in self.parameters:
            parameter.resetToPrior()

    def normalizedToPhysical(self, normalizedValues: np.ndarray) -> np.ndarray:
        self._requireParameters()
        return normalizedToPhysical(normalizedValues, self.priors, self.stepSizes)

    def physicalToNormalized(self, physicalValues: np.ndarray) -> np.ndarray:
        self._requireParameters()
        return physicalToNormalized(physicalValues, self.priors, self.stepSizes)

    def evaluateLlh(
        self,
        physicalValues: np.ndarray | None = None,
        normalizedValues: np.ndarray | None = None,
        logPath: str | os.PathLike[str] | None = None,
    ) -> float:
        with preservedWorkingDirectory():
            self._requireParameters()
            if physicalValues is not None and normalizedValues is not None:
                raise ValueError("Provide either physicalValues or normalizedValues, not both")
            if normalizedValues is not None:
                physicalValues = self.normalizedToPhysical(normalizedValues)
            if physicalValues is not None:
                self.setParameterValues(physicalValues)

            if logPath is not None:
                logPath = Path(logPath).expanduser().resolve()
            workingDirectory = Path(self.context.workDir).expanduser().resolve()

            with temporaryWorkingDirectory(workingDirectory):
                with maybeRedirectNativeOutput(logPath):
                    self.engine.getLikelihoodInterface().propagateAndEvalLikelihood()
                return float(self.engine.getLikelihoodInterface().getLastLikelihood())

    def minimize(
        self,
        logPath: str | os.PathLike[str] | None = None,
    ) -> float:
        with preservedWorkingDirectory():
            self._requireParameters()
            if logPath is not None:
                logPath = Path(logPath).expanduser().resolve()
            workingDirectory = Path(self.context.workDir).expanduser().resolve()

            with temporaryWorkingDirectory(workingDirectory):
                with maybeRedirectNativeOutput(logPath):
                    self.engine.getMinimizer().minimize()

            self.refreshParameters()
            return float(self.engine.getLikelihoodInterface().getLastLikelihood())

    def evaluatePostfitThrows(
        self,
        nThrows: int,
        logPath: str | os.PathLike[str] | None = None,
        showProgress: bool = True,
    ) -> PostfitThrowSamples:
        """Throw post-fit parameters, propagate them, and evaluate their LLH.

        The GUNDAM binding only exposes ``throwPostfitParameters()`` as a state
        update on the minimizer. This method wraps that operation into a simple
        batch interface. ``logPath`` is accepted for backward compatibility but
        is intentionally ignored: native output is not redirected in this loop.
        """
        from tqdm.auto import tqdm

        with preservedWorkingDirectory():
            self._requireParameters()
            if nThrows < 1:
                raise ValueError("nThrows must be >= 1")
            workingDirectory = Path(self.context.workDir).expanduser().resolve()

            physicalValues = np.empty((nThrows, self.priors.shape[0]), dtype=np.float64)
            normalizedValues = np.empty_like(physicalValues)
            llh = np.empty(nThrows, dtype=np.float64)

            with temporaryWorkingDirectory(workingDirectory):
                minimizer = self.engine.getMinimizer()
                likelihoodInterface = self.engine.getLikelihoodInterface()
                throwIterator = range(nThrows)
                if showProgress:
                    throwIterator = tqdm(
                        throwIterator,
                        desc="GUNDAM post-fit throws",
                        unit="throw",
                    )
                for throwIndex in throwIterator:
                    minimizer.throwPostfitParameters()
                    physicalValues[throwIndex] = self.getParameterValues()
                    normalizedValues[throwIndex] = self.physicalToNormalized(
                        physicalValues[throwIndex]
                    )
                    likelihoodInterface.propagateAndEvalLikelihood()
                    llh[throwIndex] = float(likelihoodInterface.getLastLikelihood())

            self.refreshParameters()
            return PostfitThrowSamples(
                physicalValues=physicalValues,
                normalizedValues=normalizedValues,
                llh=llh,
            )

    def setSeed(self, seed: int | None = None) -> None:
        self._requireConfigured()
        seed = self.context.randomSeed if seed is None else seed
        self._setEngineRandomSeed(self.engine, seed)

    @staticmethod
    def _setEngineRandomSeed(engine, seed: int | None) -> None:
        if seed is None:
            return
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be >= 0")
        type(engine).setRandomSeed(seed)

    def _requireConfigured(self) -> None:
        if self.engine is None:
            raise RuntimeError("GundamInterface.configure() must be called first")

    def _setLikelihoodDataType(self) -> None:
        self._requireConfigured()
        gundam = self.importGundam()
        likelihoodInterface = self.engine.getLikelihoodInterface()
        dataType = getattr(gundam.LikelihoodInterface.DataType, self.context.dataType)
        likelihoodInterface.setDataType(dataType)

    def _requireParameters(self) -> None:
        self._requireConfigured()
        if not self.parameters:
            raise RuntimeError(
                "No active parameters are loaded. Call initialize() or refreshParameters() first."
            )
