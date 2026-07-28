from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .parameters import GundamParametersManager
from .root_state import GundamRootStateReader
from .runtime import GundamRuntime
from .samples import GundamSamples
from .utils import preservedWorkingDirectory


@dataclass(frozen=True, slots=True)
class PostfitThrowSamples:
    """GUNDAM post-fit throws with propagated likelihood evaluations."""

    physicalValues: np.ndarray
    llh: np.ndarray


class GundamInterface:
    """Thin Python wrapper around the GUNDAM fitting interface."""

    def __init__(self, runtime: GundamRuntime):
        # Externals
        self._runtime = runtime

        # GUNDAM objects
        self.engine: Any | None = None

        # Interface views
        self._parametersManager: GundamParametersManager | None = None

        # Internals
        self._isConfigured = False
        self._isInitialized = False

    def getRuntime(self) -> GundamRuntime:
        return self._runtime

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

    def configure(self, validatePaths: bool = True) -> None:
        with preservedWorkingDirectory():
            if validatePaths:
                self._runtime.validatePaths()

            gundam = self._runtime.getGundamModule()
            fitterEngineConfig = self._runtime.getFitterEngineConfig()

            engine = gundam.FitterEngine()
            engine.setConfig(fitterEngineConfig)
            self._setEngineRandomSeed(engine, self._runtime.randomSeed)
            with self._runtime.runFromWorkingDirectory():
                engine.configure()

            self.engine = engine
            self._parametersManager = GundamParametersManager(
                _handle=engine.getLikelihoodInterface().getModelPropagator().getParametersManager()
            )
            self._isConfigured = True
            self._isInitialized = False

    def initialize(
        self,
        logPath: str | os.PathLike[str] | None = None,
    ) -> None:
        with preservedWorkingDirectory():
            self._requireConfigured()
            if logPath is not None:
                logPath = Path(logPath).expanduser().resolve()
            redirectContext = self._runtime.logRedirector.redirect(
                logPath,
                prefix="gundam_initialize",
            )

            with self._runtime.runFromWorkingDirectory():
                self._setLikelihoodDataType()
                with redirectContext:
                    self.engine.initialize()
                self._loadDataHistogramsIfAvailable()
                self._loadPostFitStateIfRequested()
            self._isInitialized = True

    def getParametersManager(self) -> GundamParametersManager | None:
        self._requireConfigured()
        return self._parametersManager

    def evaluateLlh(
        self,
        physicalValues: np.ndarray | None = None,
        logPath: str | os.PathLike[str] | None = None,
    ) -> float:
        with preservedWorkingDirectory():
            self._requireInitialized()
            if physicalValues is not None:
                self._parametersManager.setParameterValues(physicalValues)

            with self._runtime.runFromWorkingDirectory():
                self.engine.getLikelihoodInterface().propagateAndEvalLikelihood()
                return float(self.engine.getLikelihoodInterface().getLastLikelihood())

    def minimize(
        self,
        logPath: str | os.PathLike[str] | None = None,
    ) -> float:
        with preservedWorkingDirectory():
            self._requireInitialized()
            with self._runtime.runFromWorkingDirectory():
                self.engine.getMinimizer().minimize()

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
            self._requireInitialized()
            if nThrows < 1:
                raise ValueError("nThrows must be >= 1")
            physicalValues = np.empty((nThrows, self.priors.shape[0]), dtype=np.float64)
            llh = np.empty(nThrows, dtype=np.float64)

            with self._runtime.runFromWorkingDirectory():
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
                    physicalValues[throwIndex] = self._parametersManager.getParameterValues()
                    likelihoodInterface.propagateAndEvalLikelihood()
                    llh[throwIndex] = float(likelihoodInterface.getLastLikelihood())

            return PostfitThrowSamples(
                physicalValues=physicalValues,
                llh=llh,
            )

    def setSeed(self, seed: int | None = None) -> None:
        self._requireConfigured()
        seed = self._runtime.randomSeed if seed is None else seed
        self._setEngineRandomSeed(self.engine, seed)

    def _loadDataHistogramsIfAvailable(self) -> None:
        if self._runtime.outputRootPath is None or not self._runtime.loadDataHistograms:
            return

        stateReader = GundamRootStateReader(self._runtime.absoluteOutputRootPath)
        for sample in self.dataSamples:
            sampleName = str(sample.handle.getName())
            histogramState = stateReader.readDataHistogram(sampleName)
            binContents = sample.histogram.binContents
            if len(binContents) != histogramState.sumWeights.shape[0]:
                raise ValueError(
                    f"Mismatching bin number for data sample '{sampleName}': "
                    f"ROOT histogram has {histogramState.sumWeights.shape[0]} bins, "
                    f"GUNDAM sample has {len(binContents)} bins"
                )
            for binContent, sumWeight, sqrtSumSqWeight in zip(
                binContents,
                histogramState.sumWeights,
                histogramState.sqrtSumSqWeights,
            ):
                binContent.sumWeights = float(sumWeight)
                binContent.sqrtSumSqWeights = float(sqrtSumSqWeight)

    def _loadPostFitStateIfRequested(self) -> None:
        if not self._runtime.loadPostFitState:
            return

        gundam = self._runtime.getGundamModule()
        stateReader = GundamRootStateReader(self._runtime.absoluteOutputRootPath)
        stateConfigBuilder = stateReader.buildPostFitParameterStateConfig(gundam)
        self._requireConfigured()
        self._parametersManager.injectParametersState(stateConfigBuilder.toString())

    @staticmethod
    def _setEngineRandomSeed(engine, seed: int | None) -> None:
        if seed is None:
            return
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be >= 0")
        type(engine).setRandomSeed(seed)

    def _setLikelihoodDataType(self) -> None:
        self._requireConfigured()
        gundam = self._runtime.getGundamModule()
        likelihoodInterface = self.engine.getLikelihoodInterface()
        dataType = getattr(gundam.LikelihoodInterface.DataType, self._runtime.dataType)
        likelihoodInterface.setDataType(dataType)

    def _requireConfigured(self) -> None:
        if self.engine is None:
            raise RuntimeError("GundamInterface.configure() must be called first")

    def _requireInitialized(self) -> None:
        if not self._isInitialized:
            raise RuntimeError("GundamInterface.initialize() must be called first")
