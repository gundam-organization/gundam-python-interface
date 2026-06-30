from importlib.metadata import version

import gundam_interface


def test_package_exposes_version() -> None:
    assert gundam_interface.__version__ == version("gundam-interface")


def test_package_exposes_public_api() -> None:
    assert gundam_interface.GundamRuntime.__name__ == "GundamRuntime"
    assert gundam_interface.GundamInterface.__name__ == "GundamInterface"
    assert gundam_interface.GundamLoader.__name__ == "GundamLoader"
    assert gundam_interface.GundamParameter.__name__ == "GundamParameter"


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
        loader=gundam_interface.GundamLoader(pythonPath=tmp_path / "gundam-lib"),
    )

    data = runtime.toDict()

    assert "pythonPath" not in data
    assert data["nCpuThreads"] == 1
    assert data["loader"]["pythonPath"] == str(tmp_path / "gundam-lib")


def test_gundam_runtime_loads_legacy_python_path_into_loader(tmp_path) -> None:
    runtime = gundam_interface.GundamRuntime.fromDict(
        {
            "pythonPath": str(tmp_path / "gundam-lib"),
            "workDir": str(tmp_path),
            "configPath": "config.yaml",
        }
    )

    assert runtime.loader.pythonPath == tmp_path / "gundam-lib"
