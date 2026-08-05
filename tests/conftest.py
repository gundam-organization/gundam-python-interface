from array import array
from dataclasses import dataclass
from pathlib import Path

import pytest

import gundam_interface


def pytest_addoption(parser):
    parser.addoption(
        "--gundam-lib-path",
        action="store",
        default=None,
        help="Path to the GUNDAM installation lib directory.",
    )


@dataclass(frozen=True)
class GundamTestInputs:
    rootPath: Path
    configPath: Path


@pytest.fixture(scope="session")
def gundam_test_inputs(tmp_path_factory) -> GundamTestInputs:
    """Generate the ROOT/YAML pair shared by the numbered integration tests."""
    import uproot

    workDir = tmp_path_factory.mktemp("gundam-inputs")
    rootPath = workDir / "python-interface.root"
    configPath = workDir / "python-interface.yaml"

    with uproot.recreate(rootPath) as rootFile:
        tree = rootFile.mktree("tree_mc", {"X": "float64"})
        tree.extend({"X": array("d", [-0.5] * 10)})

    configPath.write_text(
        f"""\
fitterEngineConfig:
  likelihoodInterfaceConfig:
    jointProbabilityConfig:
      type: PoissonLLH
      ignoreBinsWithZeroPredictionAtPrior: true
    dataSetList:
      - name: TestSample
        isEnabled: true
        model:
          tree: tree_mc
          filePathList:
            - "{rootPath}"
  propagatorConfig:
    sampleSetConfig:
      sampleList:
        - name: X
          isEnabled: true
          binning:
            binningDefinition:
              - name: X
                edges: [-1, 0, 1]
          dataSets: [TestSample]
""",
        encoding="utf-8",
    )
    return GundamTestInputs(rootPath=rootPath, configPath=configPath)


@pytest.fixture(scope="session")
def configured_gundam_interface(gundam_test_inputs, request):
    """Configure and initialize one real interface for the numbered tests."""
    interface = gundam_interface.GundamInterface(
        runtime=gundam_interface.GundamRuntime(
            workDir=gundam_test_inputs.configPath.parent,
            loader=gundam_interface.GundamLoader(
                gundamLibPath=request.config.getoption("--gundam-lib-path")
            ),
            configPath=gundam_test_inputs.configPath,
            dataType="Asimov",
        ),
    )
    interface.configure()
    interface.initialize()
    return interface
