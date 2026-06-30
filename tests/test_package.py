from importlib.metadata import version

import gundam_interface


def test_package_exposes_version() -> None:
    assert gundam_interface.__version__ == version("gundam-interface")


def test_package_exposes_public_api() -> None:
    assert gundam_interface.GundamRuntime.__name__ == "GundamRuntime"
    assert gundam_interface.GundamInterface.__name__ == "GundamInterface"
    assert gundam_interface.GundamParameter.__name__ == "GundamParameter"
    assert gundam_interface.importGundam.__name__ == "importGundam"
    assert gundam_interface.setupPythonPath.__name__ == "setupPythonPath"


def test_gundam_runtime_defaults_to_one_cpu_thread(tmp_path) -> None:
    runtime = gundam_interface.GundamRuntime(
        workDir=tmp_path,
        configPath="config.yaml",
    )

    assert runtime.nCpuThreads == 1


def test_gundam_runtime_serialization_excludes_python_path(tmp_path) -> None:
    runtime = gundam_interface.GundamRuntime(
        workDir=tmp_path,
        configPath="config.yaml",
        overrideList=["override.yaml"],
    )

    data = runtime.toDict()

    assert "pythonPath" not in data
    assert data["nCpuThreads"] == 1
