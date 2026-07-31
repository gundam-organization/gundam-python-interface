from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .internal.propagator import PropagatorView
from .internal.minimizer import GundamMinimizer
from .internal.parameters import ParametersManagerView
from .internal.root_state import GundamRootStateReader
from .internal.samples import SampleSetView
from .internal.utils import preservedWorkingDirectory
from .runtime import GundamRuntime


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

        # Internals
        self._isConfigured = False
        self._isInitialized = False

    def getRuntime(self) -> GundamRuntime:
        return self._runtime

    def getMinimizer(self) -> GundamMinimizer:
        self._requireConfigured()
        return GundamMinimizer(_handle=self.engine.getMinimizer())

    def getModel(self) -> PropagatorView:
        self._requireConfigured()
        return PropagatorView(_handle=self.engine.getLikelihoodInterface().getModelPropagator())

    def getData(self) -> PropagatorView:
        self._requireConfigured()
        return PropagatorView(_handle=self.engine.getLikelihoodInterface().getDataPropagator())

    def configure(self, validatePaths: bool = True) -> None:
        with preservedWorkingDirectory():
            if validatePaths:
                self._runtime.validatePaths()

            gundam = self._runtime.getGundamModule()
            fitterEngineConfig = self._runtime.getFitterEngineConfig()

            engine = gundam.FitterEngine()
            engine.setConfig(fitterEngineConfig)
            with self._runtime.runFromWorkingDirectory():
                engine.configure()

            self.engine = engine
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

        del logPath
        with preservedWorkingDirectory():
            self._requireInitialized()
            if nThrows < 1:
                raise ValueError("nThrows must be >= 1")
            parametersManager = self.getParametersManager()
            if parametersManager is None:
                raise RuntimeError("GUNDAM parameters manager is not available")
            physicalValues = np.empty(
                (nThrows, parametersManager.getParameterValues().shape[0]),
                dtype=np.float64,
            )
            llh = np.empty(nThrows, dtype=np.float64)

            with self._runtime.runFromWorkingDirectory():
                likelihoodInterface = self.engine.getLikelihoodInterface()
                throwIterator = range(nThrows)
                if showProgress:
                    throwIterator = tqdm(
                        throwIterator,
                        desc="GUNDAM post-fit throws",
                        unit="throw",
                    )
                for throwIndex in throwIterator:
                    self.getMinimizer().throwPostfitParameters()
                    physicalValues[throwIndex] = parametersManager.getParameterValues()
                    likelihoodInterface.propagateAndEvalLikelihood()
                    llh[throwIndex] = float(likelihoodInterface.getLastLikelihood())

            return PostfitThrowSamples(
                physicalValues=physicalValues,
                llh=llh,
            )

    def _loadDataHistogramsIfAvailable(self) -> None:
        if self._runtime.outputRootPath is None or not self._runtime.loadDataHistograms:
            return

        stateReader = GundamRootStateReader(self._runtime.absoluteOutputRootPath)
        dataSamples = self.getData().getSampleSet().getSampleList()
        for sample in dataSamples:
            sampleName = sample.getName()
            histogramState = stateReader.readDataHistogram(sampleName)
            bins = sample.getHistogram().getBinList()
            if len(bins) != histogramState.sumWeights.shape[0]:
                raise ValueError(
                    f"Mismatching bin number for data sample '{sampleName}': "
                    f"ROOT histogram has {histogramState.sumWeights.shape[0]} bins, "
                    f"GUNDAM sample has {len(bins)} bins"
                )
            for histogramBin, sumWeight, sqrtSumSqWeight in zip(
                bins,
                histogramState.sumWeights,
                histogramState.sqrtSumSqWeights,
            ):
                histogramBin.setSumWeights(sumWeight)
                histogramBin.setSqrtSumSqWeights(sqrtSumSqWeight)

    def _loadPostFitStateIfRequested(self) -> None:
        if not self._runtime.loadPostFitState:
            return

        gundam = self._runtime.getGundamModule()
        stateReader = GundamRootStateReader(self._runtime.absoluteOutputRootPath)
        stateConfigBuilder = stateReader.buildPostFitParameterStateConfig(gundam)
        self._requireConfigured()
        self._parametersManager.injectParametersState(stateConfigBuilder.toString())

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
