from importlib.metadata import version
from inspect import signature

import gundam_interface
from gundam_interface.logging import GundamLogRedirector


def test_gundam_log_redirector_does_not_redirect_regular_python(monkeypatch) -> None:
    monkeypatch.setattr(gundam_interface.logging, "isNotebookRuntime", lambda: False)
    redirector = GundamLogRedirector()

    with redirector.redirect() as value:
        assert value is None


def test_package_exposes_version() -> None:
    assert gundam_interface.__version__ == version("gundam-interface")


def test_package_exposes_public_api() -> None:
    assert gundam_interface.GundamRuntime.__name__ == "GundamRuntime"
    assert gundam_interface.GundamInterface.__name__ == "GundamInterface"
    assert gundam_interface.GundamLoader.__name__ == "GundamLoader"
    assert gundam_interface.GundamParameter.__name__ == "GundamParameter"
    assert gundam_interface.GundamHistogram.__name__ == "GundamHistogram"
    assert gundam_interface.GundamSample.__name__ == "GundamSample"
    assert gundam_interface.GundamSamples.__name__ == "GundamSamples"


def test_gundam_runtime_defaults_to_one_cpu_thread(tmp_path) -> None:
    runtime = gundam_interface.GundamRuntime(
        workDir=tmp_path,
        loader=gundam_interface.GundamLoader(),
        configPath="config.yaml",
    )

    assert runtime.nCpuThreads == 1


def test_gundam_runtime_serialization_excludes_python_path(tmp_path) -> None:
    runtime = gundam_interface.GundamRuntime(
        workDir=tmp_path,
        configPath="config.yaml",
        overrideList=["override.yaml"],
        loader=gundam_interface.GundamLoader(gundamLibPath=tmp_path / "gundam-lib"),
    )

    data = runtime.toDict()

    assert "pythonPath" not in data
    assert "logRedirector" not in data
    assert data["nCpuThreads"] == 1
    assert data["loader"]["gundamLibPath"] == str(tmp_path / "gundam-lib")


def test_gundam_runtime_owns_log_redirector(tmp_path) -> None:
    runtime = gundam_interface.GundamRuntime(
        workDir=tmp_path,
        loader=gundam_interface.GundamLoader(),
        configPath="config.yaml",
    )

    assert isinstance(runtime.logRedirector, GundamLogRedirector)
    runtime.logRedirector.debug = True
    assert runtime.logRedirector.debug is True


def test_gundam_interface_methods_do_not_expose_log_debug_option() -> None:
    assert "debugLogRedirection" not in signature(gundam_interface.GundamInterface.initialize).parameters
    assert "debugLogRedirection" not in signature(gundam_interface.GundamInterface.evaluateLlh).parameters
    assert "debugLogRedirection" not in signature(gundam_interface.GundamInterface.minimize).parameters


def test_gundam_samples_exposes_histogram_sum_weights() -> None:
    samples = gundam_interface.GundamSamples(
        propagator=FakePropagator(
            [FakeSample([1.0, 2.5, 3.0]), FakeSample([4.0, 5.0])]
        ),
    )

    assert len(samples) == 2
    assert samples[0].index == 0
    assert samples[0].histogram.sumWeights.tolist() == [1.0, 2.5, 3.0]
    assert samples.sumWeights(1).tolist() == [4.0, 5.0]
    assert [weights.tolist() for weights in samples.allSumWeights()] == [
        [1.0, 2.5, 3.0],
        [4.0, 5.0],
    ]


def test_gundam_interface_exposes_model_and_data_samples(tmp_path) -> None:
    modelPropagator = FakePropagator([FakeSample([1.0])])
    dataPropagator = FakePropagator([FakeSample([2.0])])
    interface = gundam_interface.GundamInterface(
        runtime=gundam_interface.GundamRuntime(
            workDir=tmp_path,
            loader=gundam_interface.GundamLoader(),
            configPath="config.yaml",
        ),
        gundam=None,
    )
    interface.engine = FakeEngine(
        FakeLikelihoodInterface(
            modelPropagator=modelPropagator,
            dataPropagator=dataPropagator,
        )
    )

    assert interface.modelSamples.sumWeights(0).tolist() == [1.0]
    assert interface.dataSamples.sumWeights(0).tolist() == [2.0]


def test_gundam_interface_exposes_minimizer_fit_parameters(tmp_path) -> None:
    fitParameters = [object(), object()]
    interface = gundam_interface.GundamInterface(
        runtime=gundam_interface.GundamRuntime(
            workDir=tmp_path,
            loader=gundam_interface.GundamLoader(),
            configPath="config.yaml",
        ),
        gundam=None,
    )
    interface.engine = FakeEngineWithMinimizer(FakeMinimizer(fitParameters))

    assert interface.minimizerFitParameters is fitParameters


def test_gundam_runtime_loads_legacy_python_path_into_loader(tmp_path) -> None:
    runtime = gundam_interface.GundamRuntime.fromDict(
        {
            "pythonPath": str(tmp_path / "gundam-lib"),
            "workDir": str(tmp_path),
            "configPath": "config.yaml",
        }
    )

    assert runtime.loader.gundamLibPath == tmp_path / "gundam-lib"


class FakeBinContent:
    def __init__(self, sumWeights) -> None:
        self.sumWeights = sumWeights


class FakeHistogram:
    def __init__(self, sumWeights) -> None:
        self._binContentList = [
            FakeBinContent(sumWeight) for sumWeight in sumWeights
        ]

    def getBinContentList(self):
        return self._binContentList


class FakeSample:
    def __init__(self, sumWeights) -> None:
        self._histogram = FakeHistogram(sumWeights)

    def getHistogram(self):
        return self._histogram


class FakeSampleSet:
    def __init__(self, samples) -> None:
        self._samples = samples

    def getSampleList(self):
        return self._samples


class FakePropagator:
    def __init__(self, samples) -> None:
        self._sampleSet = FakeSampleSet(samples)

    def getSampleSet(self):
        return self._sampleSet


class FakeLikelihoodInterface:
    def __init__(self, modelPropagator, dataPropagator) -> None:
        self._modelPropagator = modelPropagator
        self._dataPropagator = dataPropagator

    def getModelPropagator(self):
        return self._modelPropagator

    def getDataPropagator(self):
        return self._dataPropagator


class FakeEngine:
    def __init__(self, likelihoodInterface) -> None:
        self._likelihoodInterface = likelihoodInterface

    def getLikelihoodInterface(self):
        return self._likelihoodInterface


class FakeMinimizer:
    def __init__(self, fitParameters) -> None:
        self._fitParameters = fitParameters

    def getMinimizerFitParameterPtr(self):
        return self._fitParameters


class FakeEngineWithMinimizer:
    def __init__(self, minimizer) -> None:
        self._minimizer = minimizer

    def getMinimizer(self):
        return self._minimizer


def test_gundam_runtime_loads_gundam_lib_path_into_loader(tmp_path) -> None:
    runtime = gundam_interface.GundamRuntime.fromDict(
        {
            "gundamLibPath": str(tmp_path / "gundam-lib"),
            "workDir": str(tmp_path),
            "configPath": "config.yaml",
        }
    )

    assert runtime.loader.gundamLibPath == tmp_path / "gundam-lib"
