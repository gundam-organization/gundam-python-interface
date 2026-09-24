import ctypes
import os
from types import SimpleNamespace

import pytest

from gundam_interface import GundamInterface, GundamLoader, GundamRuntime


@pytest.fixture(autouse=True)
def clearPendingNativeOutput(capfd):
    # Integration tests may leave native logs buffered across test boundaries.
    # Flush before clearing capture so they cannot leak into these assertions.
    ctypes.CDLL(None).fflush(None)
    capfd.readouterr()


@pytest.fixture
def runtime(tmp_path):
    return GundamRuntime(
        loader=GundamLoader(), workDir=tmp_path, configPath="config.yaml",
    )


def emitNativeOutput(*args):
    os.write(1, b"native stdout\n")
    os.write(2, b"native stderr\n")
    # Exercise buffered C output as well as direct descriptor writes.
    ctypes.CDLL(None).puts(b"buffered native stdout")


def test_quiet_restores_output_after_exception_and_nested_context(runtime, capfd):
    runtime.setQuiet(True)
    emitNativeOutput()
    with pytest.raises(RuntimeError, match="native failure"):
        with runtime._outputContext():
            emitNativeOutput()
            with runtime._outputContext():
                emitNativeOutput()
            emitNativeOutput()
            raise RuntimeError("native failure")
    emitNativeOutput()
    ctypes.CDLL(None).fflush(None)
    captured = capfd.readouterr()
    assert captured.out == "native stdout\nbuffered native stdout\n" * 2
    assert captured.err == "native stderr\n" * 5


@pytest.mark.parametrize(
    ("operation", "messageCount"),
    [("configure", 4), ("initialize", 4), ("evaluateLlh", 1)],
)
def test_interface_quiet_can_be_toggled(
    runtime, operation, messageCount, monkeypatch, tmp_path, capfd,
):
    interface = GundamInterface(runtime)
    likelihood = SimpleNamespace(
        propagateAndEvalLikelihood=emitNativeOutput,
        getLastLikelihood=lambda: 12.5,
    )
    engine = SimpleNamespace(
        setConfig=emitNativeOutput,
        configure=emitNativeOutput,
        initialize=emitNativeOutput,
        getLikelihoodInterface=lambda: likelihood,
    )

    def getModule(self):
        emitNativeOutput()
        return SimpleNamespace(FitterEngine=lambda: engine)

    def getConfig(self):
        emitNativeOutput()
        return object()

    monkeypatch.setattr(GundamRuntime, "getGundamModule", getModule)
    monkeypatch.setattr(GundamRuntime, "getFitterEngineConfig", getConfig)
    monkeypatch.setattr(interface, "_setLikelihoodDataType", emitNativeOutput)
    monkeypatch.setattr(interface, "_loadDataHistogramsIfAvailable", emitNativeOutput)
    monkeypatch.setattr(interface, "_loadPostFitStateIfRequested", emitNativeOutput)
    monkeypatch.setattr("gundam_interface.internal.logging.isNotebookRuntime", lambda: True)

    def noTemporaryFile(*args, **kwargs):
        pytest.fail("Quiet mode must not create a temporary log")

    monkeypatch.setattr(
        "gundam_interface.internal.logging.tempfile.NamedTemporaryFile", noTemporaryFile,
    )
    runtime.logRedirector.stream = True
    runtime.logRedirector.debug = True
    interface.engine = engine
    interface._isInitialized = True
    logPath = tmp_path / "explicit.log"
    kwargs = {"validatePaths": False} if operation == "configure" else {"logPath": logPath}

    runtime.setQuiet(True)
    result = getattr(interface, operation)(**kwargs)
    assert capfd.readouterr() == ("", "native stderr\n" * messageCount)
    assert not logPath.exists()
    if operation == "evaluateLlh":
        assert result == 12.5

    runtime.setQuiet(False)
    runtime.logRedirector.redirectNotebookOutput = False
    if operation == "initialize":
        kwargs = {}
    getattr(interface, operation)(**kwargs)
    ctypes.CDLL(None).fflush(None)
    captured = capfd.readouterr()
    assert "native stdout" in captured.out
    assert "native stderr" in captured.err


def test_quiet_initialize_failure_restores_output(runtime, monkeypatch, capfd):
    interface = GundamInterface(runtime)

    def fail():
        emitNativeOutput()
        raise RuntimeError("initialization failed")

    interface.engine = SimpleNamespace(initialize=fail)
    monkeypatch.setattr(interface, "_setLikelihoodDataType", emitNativeOutput)
    runtime.setQuiet(True)
    originalDirectory = os.getcwd()
    with pytest.raises(RuntimeError, match="initialization failed"):
        interface.initialize()
    assert os.getcwd() == originalDirectory
    assert not interface._isInitialized
    assert capfd.readouterr() == ("", "native stderr\n" * 2)
    emitNativeOutput()
    ctypes.CDLL(None).fflush(None)
    captured = capfd.readouterr()
    assert "native stdout" in captured.out
    assert "native stderr" in captured.err


def test_regular_log_redirection_still_captures_stderr(runtime, tmp_path, capfd):
    logPath = tmp_path / "native.log"
    with runtime.logRedirector.redirectNativeOutput(logPath):
        emitNativeOutput()
    assert capfd.readouterr() == ("", "")
    log = logPath.read_text()
    assert "native stdout\n" in log
    assert "buffered native stdout\n" in log
    assert "native stderr\n" in log
