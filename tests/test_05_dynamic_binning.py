import json
import math
from array import array

import pytest

from gundam_interface import GundamInterface, GundamLoader, GundamRuntime


@pytest.mark.parametrize("threads", [1, 4])
def test_modify_event_variables_and_rebin(tmp_path, request, threads):
    import uproot

    rootPath = tmp_path / "dynamic.root"
    with uproot.recreate(rootPath) as rootFile:
        tree = rootFile.mktree("events", {
            "X": "float64", "weight": "float64", "identifier": "uint64",
        })
        tree.extend({
            "X": array("d", [-0.5, 0.5, 2.0]),
            "weight": array("d", [1, 2, 3]),
            "identifier": array("Q", [2**53 + 1, 2**53 + 2, 2**53 + 3]),
        })
    binning = {"binningDefinition": [{"name": "X", "edges": [-1, 0, 1]}]}
    config = {"fitterEngineConfig": {
        "likelihoodInterfaceConfig": {
            "jointProbabilityConfig": {
                "type": "LeastSquares", "lsqPoissonianApproximation": True,
            },
            "backendManagerConfig": {"isEnabled": False},
            "dataSetList": [{"name": "inputs", "model": {
                "tree": "events", "filePathList": [str(rootPath)],
                "nominalWeightFormula": "weight",
                "additionalLeavesStorage": ["identifier"],
                "allowMultipleSamplesPerEntry": True,
            }}],
        },
        "propagatorConfig": {"sampleSetConfig": {"sampleList": [
            {"name": "dynamic", "binning": binning},
            {"name": "static", "binning": binning},
        ]}},
    }}
    runtime = GundamRuntime(
        loader=GundamLoader(gundamLibPath=request.config.getoption("--gundam-lib-path")),
        workDir=tmp_path,
        nCpuThreads=threads,
        configJsonString=json.dumps(config),
        dataType="Asimov",
    )
    runtime.setQuiet(True)
    interface = GundamInterface(runtime)
    interface.configure()
    model = interface.getModel()
    sample, static = model.getSampleSet().getSampleList()
    assert not sample.isUseDynamicBinning()
    sample.setUseDynamicBinning(True)
    assert sample.isUseDynamicBinning()
    assert not static.isUseDynamicBinning()
    interface.initialize()

    events = sample.getEventList()
    assert len(events) == 3
    assert len(static.getEventList()) == 2
    histogram = sample.getHistogram()
    # Hold onto views to verify they stay valid across migrations.
    bins = histogram.getBinList()

    def check(expectedIndices, sums, errors):
        interface.evaluateLlh()
        assert [event.getIndices().bin for event in events] == expectedIndices
        assert sample.getNbBinnedEvents() == sum(index != -1 for index in expectedIndices)
        assert [bin.getSumWeights() for bin in bins] == pytest.approx(sums)
        assert [bin.getSqrtSumSqWeights() for bin in bins] == pytest.approx(errors)
        assert [bin.getSumWeights() for bin in static.getHistogram().getBinList()] == [1, 2]
        dataSample = interface.getData().getSampleSet().getSampleList()[0]
        assert [bin.getSumWeights() for bin in dataSample.getHistogram().getBinList()] == [1, 2]

    check([0, 1, -1], [1, 2], [1, 2])
    variables = events[0].getVariables()
    assert variables["identifier"] == 2**53 + 1
    variables["identifier"] = 2**63 + 123
    assert events[0].getVariables()["identifier"] == 2**63 + 123
    assert isinstance(variables["identifier"], int)
    xIndex = list(variables.toDict()).index("X")
    variables["X"] = 0.25
    events[2].getVariables()[xIndex] = 0.0
    # Editing variables alone does not rebin events during evaluation.
    check([0, 1, -1], [1, 2], [1, 2])
    model.updateEventBinning()
    check([1, 1, 1], [0, 6], [0, math.sqrt(14)])

    events[1].getVariables()["X"] = 1.0  # Upper edge is excluded.
    sample.updateEventBinning()
    check([1, -1, 1], [0, 4], [0, math.sqrt(10)])

    for event, value in zip(events, [-0.5, 0.5, 2.0]):
        event.getVariables()["X"] = value
    for _ in range(2):
        model.updateEventBinning()
        check([0, 1, -1], [1, 2], [1, 2])
