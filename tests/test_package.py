from importlib.metadata import version
from inspect import signature
from contextlib import contextmanager
import sys

import pytest

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


def test_gundam_runtime_serializes_output_root_state(tmp_path) -> None:
    runtime = gundam_interface.GundamRuntime(
        workDir=tmp_path,
        configPath="config.yaml",
        outputRootPath="fit.root",
        loadDataHistograms=False,
        loadPostFitState=True,
        dataType="Toy",
        randomSeed=12345,
        loader=gundam_interface.GundamLoader(),
    )

    data = runtime.toDict()
    restored = gundam_interface.GundamRuntime.fromDict(data)

    assert data["configPath"] == "config.yaml"
    assert data["outputRootPath"] == "fit.root"
    assert data["loadDataHistograms"] is False
    assert data["loadPostFitState"] is True
    assert data["dataType"] == "Toy"
    assert data["randomSeed"] == 12345
    assert restored.configPath == runtime.configPath
    assert restored.outputRootPath == runtime.outputRootPath
    assert restored.loadDataHistograms is False
    assert restored.loadPostFitState is True


def test_gundam_runtime_accepts_output_root_as_config_source(tmp_path) -> None:
    runtime = gundam_interface.GundamRuntime(
        workDir=tmp_path,
        outputRootPath="fit.root",
        loader=gundam_interface.GundamLoader(),
    )

    assert runtime.absoluteOutputRootPath == tmp_path / "fit.root"


def test_gundam_runtime_rejects_load_postfit_without_output_root(tmp_path) -> None:
    with pytest.raises(ValueError, match="outputRootPath"):
        gundam_interface.GundamRuntime(
            workDir=tmp_path,
            configPath="config.yaml",
            loadPostFitState=True,
            loader=gundam_interface.GundamLoader(),
        )


def test_gundam_runtime_rejects_config_path_and_json_string(tmp_path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        gundam_interface.GundamRuntime(
            workDir=tmp_path,
            configPath="config.yaml",
            configJsonString="{}",
            loader=gundam_interface.GundamLoader(),
        )


def test_gundam_runtime_rejects_config_only_root_without_data_type(tmp_path) -> None:
    with pytest.raises(ValueError, match="dataType"):
        gundam_interface.GundamRuntime(
            workDir=tmp_path,
            outputRootPath="fit.root",
            loadDataHistograms=False,
            loader=gundam_interface.GundamLoader(),
        )


def test_gundam_runtime_rejects_config_only_toy_root_without_seed(tmp_path) -> None:
    with pytest.raises(ValueError, match="randomSeed"):
        gundam_interface.GundamRuntime(
            workDir=tmp_path,
            outputRootPath="fit.root",
            loadDataHistograms=False,
            dataType="Toy",
            loader=gundam_interface.GundamLoader(),
        )


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


def test_evaluate_llh_does_not_auto_redirect_without_log_path(tmp_path) -> None:
    likelihoodInterface = FakeEvaluatingLikelihoodInterface(llh=12.5)
    interface = makeConfiguredInterface(tmp_path, FakeEngine(likelihoodInterface))
    redirector = RecordingRedirector()
    interface.runtime.logRedirector = redirector

    llh = interface.evaluateLlh()

    assert llh == 12.5
    assert likelihoodInterface.evaluationCount == 1
    assert redirector.calls == []


def test_evaluate_llh_does_not_redirect_with_log_path(tmp_path) -> None:
    likelihoodInterface = FakeEvaluatingLikelihoodInterface(llh=12.5)
    interface = makeConfiguredInterface(tmp_path, FakeEngine(likelihoodInterface))
    redirector = RecordingRedirector()
    interface.runtime.logRedirector = redirector

    llh = interface.evaluateLlh(logPath=tmp_path / "evaluate.log")

    assert llh == 12.5
    assert likelihoodInterface.evaluationCount == 1
    assert redirector.calls == []


def test_minimize_does_not_auto_redirect_without_log_path(tmp_path) -> None:
    likelihoodInterface = FakeEvaluatingLikelihoodInterface(llh=4.0)
    minimizer = FakeRecordingMinimizer()
    interface = makeConfiguredInterface(
        tmp_path,
        FakeEngineWithLikelihoodAndMinimizer(likelihoodInterface, minimizer),
    )
    interface.refreshParameters = lambda: interface.parameters
    redirector = RecordingRedirector()
    interface.runtime.logRedirector = redirector

    llh = interface.minimize()

    assert llh == 4.0
    assert minimizer.minimizeCount == 1
    assert redirector.calls == []


def test_minimize_does_not_redirect_with_log_path(tmp_path) -> None:
    likelihoodInterface = FakeEvaluatingLikelihoodInterface(llh=4.0)
    minimizer = FakeRecordingMinimizer()
    interface = makeConfiguredInterface(
        tmp_path,
        FakeEngineWithLikelihoodAndMinimizer(likelihoodInterface, minimizer),
    )
    interface.refreshParameters = lambda: interface.parameters
    redirector = RecordingRedirector()
    interface.runtime.logRedirector = redirector

    llh = interface.minimize(logPath=tmp_path / "minimize.log")

    assert llh == 4.0
    assert minimizer.minimizeCount == 1
    assert redirector.calls == []


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


def test_gundam_histogram_exposes_2d_layout_helpers() -> None:
    histogram = gundam_interface.GundamHistogram(
        handle=FakeHistogram(
            [10.0, 20.0],
            binContexts=[
                FakeBinContext(
                    FakeBin(
                        0,
                        [
                            FakeBinEdge("CosThetamu", -1.0, 0.0),
                            FakeBinEdge("Pmu", 0.0, 320.0),
                            FakeBinEdge("SelectedSample", 131, 131, True),
                        ],
                    )
                ),
                FakeBinContext(
                    FakeBin(
                        1,
                        [
                            FakeBinEdge("CosThetamu", 0.0, 1.0),
                            FakeBinEdge("Pmu", 0.0, 320.0),
                            FakeBinEdge("SelectedSample", 131, 131, True),
                        ],
                    )
                ),
            ],
        )
    )

    assert histogram.variableNames() == ["CosThetamu", "Pmu"]
    assert histogram.variableNames(preferredOrder=("Pmu", "CosThetamu")) == [
        "Pmu",
        "CosThetamu",
    ]

    binDefinitions = histogram.binDefinitions(variableOrder=("Pmu", "CosThetamu"))
    assert binDefinitions == [
        {
            "index": 0,
            "edges": {
                "Pmu": {"min": 0.0, "max": 320.0},
                "CosThetamu": {"min": -1.0, "max": 0.0},
            },
        },
        {
            "index": 1,
            "edges": {
                "Pmu": {"min": 0.0, "max": 320.0},
                "CosThetamu": {"min": 0.0, "max": 1.0},
            },
        },
    ]

    layout = histogram.layout2d(preferredOrder=("Pmu", "CosThetamu"))
    assert layout["variable_names"] == ["Pmu", "CosThetamu"]
    assert layout["sum_weights"].tolist() == [10.0, 20.0]
    assert layout["bin_volumes"].tolist() == [320.0, 320.0]
    assert layout["values"].tolist() == [10.0, 20.0]
    assert layout["values_label"] == "sumWeights"
    assert layout["x_edges"].tolist() == [0.0, 320.0]
    assert layout["y_edges"].tolist() == [-1.0, 0.0, 1.0]
    assert layout["bins"] == [
        {
            "index": 0,
            "x_min": 0.0,
            "x_max": 320.0,
            "y_min": -1.0,
            "y_max": 0.0,
            "measure": 320.0,
            "sum_weights": 10.0,
        },
        {
            "index": 1,
            "x_min": 0.0,
            "x_max": 320.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "measure": 320.0,
            "sum_weights": 20.0,
        },
    ]

    densityLayout = histogram.layout2d(
        preferredOrder=("Pmu", "CosThetamu"),
        divideByBinVolume=True,
    )
    assert densityLayout["values_label"] == "sumWeights / binMeasure"
    assert densityLayout["values"].tolist() == [0.03125, 0.0625]


def test_gundam_histogram_projects_3d_layout_to_2d() -> None:
    histogram = gundam_interface.GundamHistogram(
        handle=FakeHistogram(
            [10.0, 20.0, 30.0, 40.0],
            binContexts=[
                FakeBinContext(
                    FakeBin(
                        0,
                        [
                            FakeBinEdge("CosThetamu", -1.0, 0.0),
                            FakeBinEdge("Pmu", 0.0, 320.0),
                            FakeBinEdge("Enu", 0.0, 1.0),
                        ],
                    )
                ),
                FakeBinContext(
                    FakeBin(
                        1,
                        [
                            FakeBinEdge("CosThetamu", -1.0, 0.0),
                            FakeBinEdge("Pmu", 0.0, 320.0),
                            FakeBinEdge("Enu", 1.0, 2.0),
                        ],
                    )
                ),
                FakeBinContext(
                    FakeBin(
                        2,
                        [
                            FakeBinEdge("CosThetamu", 0.0, 1.0),
                            FakeBinEdge("Pmu", 0.0, 320.0),
                            FakeBinEdge("Enu", 0.0, 1.0),
                        ],
                    )
                ),
                FakeBinContext(
                    FakeBin(
                        3,
                        [
                            FakeBinEdge("CosThetamu", 0.0, 1.0),
                            FakeBinEdge("Pmu", 0.0, 320.0),
                            FakeBinEdge("Enu", 1.0, 2.0),
                        ],
                    )
                ),
            ],
        )
    )

    assert histogram.binMeasures(variableOrder=("Pmu", "CosThetamu")).tolist() == [
        320.0,
        320.0,
        320.0,
        320.0,
    ]

    layout = histogram.layout2d(
        preferredOrder=("Pmu", "CosThetamu"),
        divideByBinVolume=True,
    )
    assert layout["variable_names"] == ["Pmu", "CosThetamu"]
    assert layout["sum_weights"].tolist() == [30.0, 70.0]
    assert layout["bin_volumes"].tolist() == [320.0, 320.0]
    assert layout["values"].tolist() == [0.09375, 0.21875]
    assert layout["bins"] == [
        {
            "index": 0,
            "x_min": 0.0,
            "x_max": 320.0,
            "y_min": -1.0,
            "y_max": 0.0,
            "measure": 320.0,
            "sum_weights": 30.0,
        },
        {
            "index": 1,
            "x_min": 0.0,
            "x_max": 320.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "measure": 320.0,
            "sum_weights": 70.0,
        },
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


def test_build_config_builder_prefers_config_path_over_output_root(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text("config", encoding="utf-8")
    (tmp_path / "fit.root").write_text("root", encoding="utf-8")
    (tmp_path / "override.yaml").write_text("override", encoding="utf-8")
    fakeGundam = FakeGundamModule()
    interface = gundam_interface.GundamInterface(
        runtime=gundam_interface.GundamRuntime(
            workDir=tmp_path,
            loader=gundam_interface.GundamLoader(),
            configPath="config.yaml",
            outputRootPath="fit.root",
            overrideList=["override.yaml"],
        ),
        gundam=fakeGundam,
    )

    configBuilder = interface._buildConfigBuilder(fakeGundam)

    assert configBuilder.source == str((tmp_path / "config.yaml").resolve())
    assert configBuilder.overrides == [str((tmp_path / "override.yaml").resolve())]


def test_build_config_builder_uses_output_root_when_no_config_path(tmp_path) -> None:
    (tmp_path / "fit.root").write_text("root", encoding="utf-8")
    fakeGundam = FakeGundamModule()
    interface = gundam_interface.GundamInterface(
        runtime=gundam_interface.GundamRuntime(
            workDir=tmp_path,
            loader=gundam_interface.GundamLoader(),
            outputRootPath="fit.root",
        ),
        gundam=fakeGundam,
    )

    configBuilder = interface._buildConfigBuilder(fakeGundam)

    assert configBuilder.source == str((tmp_path / "fit.root").resolve())


def test_initialize_loads_postfit_state_when_requested(tmp_path, monkeypatch) -> None:
    outputRootPath = tmp_path / "fit.root"
    outputRootPath.write_text("root", encoding="utf-8")
    fakeParametersManager = FakeInjectingParametersManager()
    installFakeUproot(
        monkeypatch,
        {
            "FitterEngine/postFit/Migrad/parameterStateAfterMinimize_TNamed": FakeTNamed(
                '{"parameterSetList":[]}'
            ),
            "FitterEngine/preFit/data/sample0/histogram": FakeTH1D([10.0, 20.0], [1.0, 2.0]),
        },
    )
    fakeGundam = FakeGundamModule()
    interface = gundam_interface.GundamInterface(
        runtime=gundam_interface.GundamRuntime(
            workDir=tmp_path,
            loader=gundam_interface.GundamLoader(),
            configPath="config.yaml",
            outputRootPath=outputRootPath,
            loadPostFitState=True,
        ),
        gundam=fakeGundam,
    )
    interface.engine = FakeInitializableEngine(fakeParametersManager)
    interface.refreshParameters = lambda: interface.parameters

    interface.initialize()

    assert interface.engine.initializeCount == 1
    assert interface.dataSamples.sumWeights(0).tolist() == [10.0, 20.0]
    assert len(fakeParametersManager.injectedConfigs) == 1
    assert fakeParametersManager.injectedConfigs[0].startswith("config:")


def test_initialize_restores_data_histograms_from_output_root_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    outputRootPath = tmp_path / "fit.root"
    outputRootPath.write_text("root", encoding="utf-8")
    fakeParametersManager = FakeInjectingParametersManager()
    installFakeUproot(
        monkeypatch,
        {
            "FitterEngine/preFit/data/sample0/histogram": FakeTH1D([3.0, 4.0], [0.3, 0.4]),
        },
    )
    fakeGundam = FakeGundamModule()
    interface = gundam_interface.GundamInterface(
        runtime=gundam_interface.GundamRuntime(
            workDir=tmp_path,
            loader=gundam_interface.GundamLoader(),
            configPath="config.yaml",
            outputRootPath=outputRootPath,
        ),
        gundam=fakeGundam,
    )
    interface.engine = FakeInitializableEngine(fakeParametersManager)
    interface.refreshParameters = lambda: interface.parameters

    interface.initialize()

    assert fakeParametersManager.injectedConfigs == []
    assert interface.dataSamples.sumWeights(0).tolist() == [3.0, 4.0]
    assert interface.dataSamples[0].histogram.sqrtSumSqWeights.tolist() == [0.3, 0.4]


def test_initialize_can_skip_data_histograms_from_output_root(tmp_path) -> None:
    outputRootPath = tmp_path / "fit.root"
    outputRootPath.write_text("root", encoding="utf-8")
    fakeParametersManager = FakeInjectingParametersManager()
    fakeGundam = FakeGundamModule()
    interface = gundam_interface.GundamInterface(
        runtime=gundam_interface.GundamRuntime(
            workDir=tmp_path,
            loader=gundam_interface.GundamLoader(),
            configPath="config.yaml",
            outputRootPath=outputRootPath,
            loadDataHistograms=False,
            dataType="Toy",
            randomSeed=12345,
        ),
        gundam=fakeGundam,
    )
    interface.engine = FakeInitializableEngine(fakeParametersManager)
    interface.refreshParameters = lambda: interface.parameters

    interface.initialize()

    assert interface.dataSamples.sumWeights(0).tolist() == [0.0, 0.0]
    assert fakeParametersManager.injectedConfigs == []


def test_initialize_fails_when_requested_postfit_state_is_missing(tmp_path, monkeypatch) -> None:
    outputRootPath = tmp_path / "fit.root"
    outputRootPath.write_text("root", encoding="utf-8")
    installFakeUproot(
        monkeypatch,
        {
            "FitterEngine/preFit/data/sample0/histogram": FakeTH1D([1.0, 2.0], [0.1, 0.2]),
        },
    )
    fakeGundam = FakeGundamModule()
    interface = gundam_interface.GundamInterface(
        runtime=gundam_interface.GundamRuntime(
            workDir=tmp_path,
            loader=gundam_interface.GundamLoader(),
            configPath="config.yaml",
            outputRootPath=outputRootPath,
            loadPostFitState=True,
        ),
        gundam=fakeGundam,
    )
    interface.engine = FakeInitializableEngine(FakeInjectingParametersManager())
    interface.refreshParameters = lambda: interface.parameters

    with pytest.raises(KeyError, match="parameterStateAfterMinimize_TNamed"):
        interface.initialize()


def test_initialize_fails_when_saved_data_histogram_bin_count_mismatches(
    tmp_path,
    monkeypatch,
) -> None:
    outputRootPath = tmp_path / "fit.root"
    outputRootPath.write_text("root", encoding="utf-8")
    installFakeUproot(
        monkeypatch,
        {
            "FitterEngine/preFit/data/sample0/histogram": FakeTH1D([1.0], [0.1]),
        },
    )
    fakeGundam = FakeGundamModule()
    interface = gundam_interface.GundamInterface(
        runtime=gundam_interface.GundamRuntime(
            workDir=tmp_path,
            loader=gundam_interface.GundamLoader(),
            configPath="config.yaml",
            outputRootPath=outputRootPath,
        ),
        gundam=fakeGundam,
    )
    interface.engine = FakeInitializableEngine(FakeInjectingParametersManager())
    interface.refreshParameters = lambda: interface.parameters

    with pytest.raises(ValueError, match="Mismatching bin number"):
        interface.initialize()


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
    def __init__(self, sumWeights, sqrtSumSqWeights=0.0) -> None:
        self.sumWeights = sumWeights
        self.sqrtSumSqWeights = sqrtSumSqWeights


class FakeBinEdge:
    def __init__(self, varName, minValue, maxValue, isConditionVar=False) -> None:
        self.varName = varName
        self.min = minValue
        self.max = maxValue
        self.isConditionVar = isConditionVar


class FakeBin:
    def __init__(self, index, edges) -> None:
        self._index = index
        self._edges = edges

    def getIndex(self):
        return self._index

    def getEdgesList(self):
        return self._edges


class FakeBinContext:
    def __init__(self, binHandle) -> None:
        self.bin = binHandle


class FakeHistogram:
    def __init__(self, sumWeights, sqrtSumSqWeights=None, binContexts=None) -> None:
        if sqrtSumSqWeights is None:
            sqrtSumSqWeights = [0.0 for _ in sumWeights]
        self._binContentList = [
            FakeBinContent(sumWeight, sqrtSumSqWeight)
            for sumWeight, sqrtSumSqWeight in zip(sumWeights, sqrtSumSqWeights)
        ]
        self._binContextList = [] if binContexts is None else list(binContexts)

    def getBinContentList(self):
        return self._binContentList

    def getBinContextList(self):
        return self._binContextList


class FakeSample:
    def __init__(
        self, sumWeights, name="sample0", sqrtSumSqWeights=None, binContexts=None
    ) -> None:
        self._name = name
        self._histogram = FakeHistogram(sumWeights, sqrtSumSqWeights, binContexts)

    def getName(self):
        return self._name

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


class FakeConfigBuilder:
    def __init__(self, source=None) -> None:
        self.source = source
        self.overrides = []

    def override(self, path) -> None:
        self.overrides.append(path)

    def getConfig(self):
        return f"config:{self.source or ''}"

    def toString(self):
        return f"config-string:{self.source or ''}"


class FakeConfigUtils:
    ConfigBuilder = FakeConfigBuilder


class FakeDataType:
    Asimov = "Asimov"
    Toy = "Toy"
    RealData = "RealData"


class FakeGundamLikelihoodInterfaceNamespace:
    DataType = FakeDataType


class FakeTNamed:
    def __init__(self, title) -> None:
        self._title = title

    def member(self, name):
        if name != "fTitle":
            raise KeyError(name)
        return self._title


class FakeTH1D:
    def __init__(self, values, errors) -> None:
        self._values = values
        self._errors = errors

    def values(self, flow=False):
        return self._values

    def errors(self, flow=False):
        return self._errors


class FakeUprootFile:
    def __init__(self, objects) -> None:
        self._objects = objects

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __getitem__(self, objectPath):
        return self._objects[objectPath]


class FakeUprootModule:
    def __init__(self, objects) -> None:
        self._objects = objects

    def open(self, path):
        return FakeUprootFile(self._objects)


def installFakeUproot(monkeypatch, objects) -> None:
    monkeypatch.setitem(sys.modules, "uproot", FakeUprootModule(objects))


class FakeGundamModule:
    ConfigUtils = FakeConfigUtils
    LikelihoodInterface = FakeGundamLikelihoodInterfaceNamespace

    def __init__(self) -> None:
        pass


class FakeInjectingParametersManager:
    def __init__(self) -> None:
        self.injectedConfigs = []

    def injectParameterValues(self, config) -> None:
        self.injectedConfigs.append(config)


class FakeStatePropagator:
    def __init__(self, parametersManager) -> None:
        self._parametersManager = parametersManager

    def getParametersManager(self):
        return self._parametersManager


class FakeStateLikelihoodInterface:
    def __init__(self, parametersManager, dataSamples) -> None:
        self._modelPropagator = FakeStatePropagator(parametersManager)
        self._dataPropagator = FakePropagator(dataSamples)
        self.dataType = None

    def setDataType(self, dataType) -> None:
        self.dataType = dataType

    def getModelPropagator(self):
        return self._modelPropagator

    def getDataPropagator(self):
        return self._dataPropagator


class FakeInitializableEngine:
    def __init__(self, parametersManager, dataSamples=None) -> None:
        if dataSamples is None:
            dataSamples = [FakeSample([0.0, 0.0], name="sample0")]
        self._likelihoodInterface = FakeStateLikelihoodInterface(
            parametersManager,
            dataSamples,
        )
        self.initializeCount = 0

    def initialize(self) -> None:
        self.initializeCount += 1

    def getLikelihoodInterface(self):
        return self._likelihoodInterface


class FakeEvaluatingLikelihoodInterface:
    def __init__(self, llh) -> None:
        self._llh = llh
        self.evaluationCount = 0

    def propagateAndEvalLikelihood(self) -> None:
        self.evaluationCount += 1

    def getLastLikelihood(self):
        return self._llh


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


class FakeRecordingMinimizer:
    def __init__(self) -> None:
        self.minimizeCount = 0

    def minimize(self) -> None:
        self.minimizeCount += 1


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


class FakeEngineWithLikelihoodAndMinimizer:
    def __init__(self, likelihoodInterface, minimizer) -> None:
        self._likelihoodInterface = likelihoodInterface
        self._minimizer = minimizer

    def getLikelihoodInterface(self):
        return self._likelihoodInterface

    def getMinimizer(self):
        return self._minimizer


class RecordingRedirector:
    def __init__(self) -> None:
        self.calls = []

    @contextmanager
    def redirect(self, logPath=None, *, prefix="gundam", stream=None):
        self.calls.append((logPath, prefix))
        yield


def makeConfiguredInterface(tmp_path, engine):
    interface = gundam_interface.GundamInterface(
        runtime=gundam_interface.GundamRuntime(
            workDir=tmp_path,
            loader=gundam_interface.GundamLoader(),
            configPath="config.yaml",
        ),
        gundam=None,
    )
    interface.engine = engine
    interface.parameters = [object()]
    return interface


def test_gundam_runtime_loads_gundam_lib_path_into_loader(tmp_path) -> None:
    runtime = gundam_interface.GundamRuntime.fromDict(
        {
            "gundamLibPath": str(tmp_path / "gundam-lib"),
            "workDir": str(tmp_path),
            "configPath": "config.yaml",
        }
    )

    assert runtime.loader.gundamLibPath == tmp_path / "gundam-lib"
