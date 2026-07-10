from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .parameters import (
    GundamParameter,
    collectActiveParameters,
    parameterPriors,
    parameterSteps,
    parameterThrowValues,
)
from .root_state import GundamRootStateReader
from .runtime import GundamRuntime
from .samples import GundamSamples


@dataclass(frozen=True, slots=True)
class PostfitThrowSamples:
    """GUNDAM post-fit throws with propagated likelihood evaluations."""

    physicalValues: np.ndarray
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

    def __init__(self, runtime: GundamRuntime, gundam: Any | None = None):
        self.runtime = runtime
        self.gundam: Any | None = gundam
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

    @property
    def modelSamples(self) -> GundamSamples:
        self._requireConfigured()
        propagator = self.engine.getLikelihoodInterface().getModelPropagator()
        return GundamSamples(propagator=propagator)

    @property
    def dataSamples(self) -> GundamSamples:
        self._requireConfigured()
        propagator = self.engine.getLikelihoodInterface().getDataPropagator()
        return GundamSamples(propagator=propagator)

    @property
    def minimizerFitParameters(self):
        self._requireConfigured()
        return self.engine.getMinimizer().getMinimizerFitParameterPtr()

    def importGundam(self):
        if self.gundam is None:
            self.gundam = self.runtime.loader.importGundam()
        return self.gundam

    def configure(self, validatePaths: bool = True) -> None:
        with preservedWorkingDirectory():
            if validatePaths:
                self.runtime.validatePaths()

            gundam = self.importGundam()
            gundam.setLightOutputMode(False)
            gundam.setNumberOfThreads(self.runtime.nCpuThreads)
            workingDirectory = Path(self.runtime.workDir).expanduser().resolve()
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
            self._setEngineRandomSeed(engine, self.runtime.randomSeed)
            with temporaryWorkingDirectory(workingDirectory):
                engine.configure()

            self.configBuilder = configBuilder
            self.configJsonString = configJsonString
            self.fitterEngineConfig = fitterEngineConfig
            self.engine = engine

    def _buildConfigBuilder(self, gundam):
        if self.runtime.configJsonString is not None:
            configBuilder = self._buildConfigBuilderFromJsonString(
                gundam,
                self.runtime.configJsonString,
            )
        elif self.runtime.configPath is not None:
            configPath = Path(self.runtime.absoluteConfigPath).expanduser().resolve()
            configBuilder = gundam.ConfigUtils.ConfigBuilder(str(configPath))
        else:
            outputRootPath = Path(self.runtime.absoluteOutputRootPath).expanduser().resolve()
            configBuilder = gundam.ConfigUtils.ConfigBuilder(str(outputRootPath))

        overridePaths = [
            Path(overridePath).expanduser().resolve()
            for overridePath in self.runtime.absoluteOverridePaths
        ]
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
            workingDirectory = Path(self.runtime.workDir).expanduser().resolve()

            if logPath is not None:
                logPath = Path(logPath).expanduser().resolve()
            redirectContext = self.runtime.logRedirector.redirect(
                logPath,
                prefix="gundam_initialize",
            )

            with temporaryWorkingDirectory(workingDirectory):
                self._setLikelihoodDataType()
                with redirectContext:
                    self.engine.initialize()
                self._loadPostFitStateIfRequested()

            self.refreshParameters()

    def _loadPostFitStateIfRequested(self) -> None:
        if not self.runtime.loadPostFitState:
            return

        gundam = self.importGundam()
        stateReader = GundamRootStateReader(self.runtime.absoluteOutputRootPath)
        stateConfigBuilder = stateReader.buildPostFitParameterStateConfig(gundam)
        parametersManager = (
            self.engine.getLikelihoodInterface()
            .getModelPropagator()
            .getParametersManager()
        )
        parametersManager.injectParameterValues(stateConfigBuilder.getConfig())

    def refreshParameters(self) -> list[GundamParameter]:
        self._requireConfigured()
        parametersManager = (
            self.engine.getLikelihoodInterface()
            .getModelPropagator()
            .getParametersManager()
        )
        self.parameters = collectActiveParameters(
            parametersManager,
            includeThrowValues=self.runtime.dataType == "Toy",
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

    def evaluateLlh(
        self,
        physicalValues: np.ndarray | None = None,
        logPath: str | os.PathLike[str] | None = None,
    ) -> float:
        with preservedWorkingDirectory():
            self._requireParameters()
            if physicalValues is not None:
                self.setParameterValues(physicalValues)

            workingDirectory = Path(self.runtime.workDir).expanduser().resolve()

            with temporaryWorkingDirectory(workingDirectory):
                self.engine.getLikelihoodInterface().propagateAndEvalLikelihood()
                return float(self.engine.getLikelihoodInterface().getLastLikelihood())

    def minimize(
        self,
        logPath: str | os.PathLike[str] | None = None,
    ) -> float:
        with preservedWorkingDirectory():
            self._requireParameters()
            workingDirectory = Path(self.runtime.workDir).expanduser().resolve()

            with temporaryWorkingDirectory(workingDirectory):
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
            workingDirectory = Path(self.runtime.workDir).expanduser().resolve()

            physicalValues = np.empty((nThrows, self.priors.shape[0]), dtype=np.float64)
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
                    likelihoodInterface.propagateAndEvalLikelihood()
                    llh[throwIndex] = float(likelihoodInterface.getLastLikelihood())

            self.refreshParameters()
            return PostfitThrowSamples(
                physicalValues=physicalValues,
                llh=llh,
            )

    def setSeed(self, seed: int | None = None) -> None:
        self._requireConfigured()
        seed = self.runtime.randomSeed if seed is None else seed
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
        dataType = getattr(gundam.LikelihoodInterface.DataType, self.runtime.dataType)
        likelihoodInterface.setDataType(dataType)

    def _requireParameters(self) -> None:
        self._requireConfigured()
        if not self.parameters:
            raise RuntimeError(
                "No active parameters are loaded. Call initialize() or refreshParameters() first."
            )
