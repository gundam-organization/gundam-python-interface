from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


def isNotebookRuntime() -> bool:
    if "ipykernel" in sys.modules:
        return True

    try:
        from IPython import get_ipython
    except ModuleNotFoundError:
        return False

    shell = get_ipython()
    if shell is None:
        return False
    if shell.__class__.__name__ == "ZMQInteractiveShell":
        return True

    config = getattr(shell, "config", {})
    return "IPKernelApp" in config


@dataclass(slots=True)
class GundamLogRedirector:
    """Policy object for GUNDAM native stdout/stderr redirection.

    Regular Python scripts keep native output untouched by default, so GUNDAM
    ``cout`` and ``cerr`` are displayed live. Notebook runtimes redirect native
    output through a temporary file by default because direct C/C++ output can
    bypass or confuse frontend capture.

    Parameters
    ----------
    redirectNotebookOutput:
        Redirect native output through an auto-deleted temporary file when
        running in a notebook and no explicit log path is provided.
    stream:
        Stream redirected output while the GUNDAM call is running. If false,
        temporary notebook logs are printed after the call completes.
    debug:
        Print redirection diagnostics in notebooks.
    """

    redirectNotebookOutput: bool = True
    stream: bool = False
    debug: bool = False

    def redirect(
        self,
        logPath: str | os.PathLike[str] | None = None,
        *,
        prefix: str = "gundam",
        stream: bool | None = None,
        debug: bool | None = None,
    ):
        """Return a context manager for the configured redirection policy."""
        stream = self.stream if stream is None else stream
        debug = self.debug if debug is None else debug

        if logPath is None:
            if not (self.redirectNotebookOutput and isNotebookRuntime()):
                return nullcontext()
            return temporaryRedirectNativeOutput(prefix=prefix, stream=stream, debug=debug)

        return redirectNativeOutput(logPath, stream=stream, debug=debug)

@contextmanager
def redirectNativeOutput(
    logPath: str | os.PathLike[str],
    *,
    stream: bool = False,
    debug: bool = False,
) -> Iterator[None]:
    """Redirect C/C++ stdout and stderr to a file.

    This is useful in Jupyter where native C++ loggers can interact poorly with
    ipykernel stdout capture. Python stdout/stderr are restored after the block.
    When ``stream`` is true, new log content is also printed as it is written.
    """
    path = Path(logPath).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if debug and isNotebookRuntime():
        print(
            f"GUNDAM native output is redirected to log file: {path}",
            flush=True,
        )

    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    sys.stderr.flush()
    libc.fflush(None)

    stdoutFd = os.dup(1)
    stderrFd = os.dup(2)
    streamStopEvent: threading.Event | None = None
    streamThread: threading.Thread | None = None

    try:
        with path.open("ab", buffering=0) as nativeLog:
            streamOffset = nativeLog.tell()
            if stream:
                streamStopEvent = threading.Event()
                streamThread = threading.Thread(
                    target=_streamLogFile,
                    args=(path, streamStopEvent, streamOffset, stdoutFd),
                    daemon=True,
                )
                streamThread.start()
            os.dup2(nativeLog.fileno(), 1)
            os.dup2(nativeLog.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        libc.fflush(None)
        if streamStopEvent is not None:
            streamStopEvent.set()
        if streamThread is not None:
            streamThread.join(timeout=2.0)
        os.dup2(stdoutFd, 1)
        os.dup2(stderrFd, 2)
        os.close(stdoutFd)
        os.close(stderrFd)


def maybeRedirectNativeOutput(
    logPath: str | os.PathLike[str] | None,
    *,
    stream: bool = False,
    prefix: str = "gundam",
    debug: bool = False,
):
    """Redirect native output only when needed.

    With ``logPath=None``, regular Python scripts keep native stdout/stderr
    untouched. Notebook runtimes redirect through a temporary file so native
    output is captured reliably by the frontend.
    """
    return GundamLogRedirector(stream=stream, debug=debug).redirect(
        logPath,
        prefix=prefix,
    )


@contextmanager
def temporaryRedirectNativeOutput(
    prefix: str = "gundam",
    *,
    stream: bool = False,
    debug: bool = False,
) -> Iterator[None]:
    """Redirect native output to a temporary log file and delete it afterwards."""
    with tempfile.NamedTemporaryFile(prefix=f"{prefix}_", suffix=".log", delete=False) as logFile:
        logPath = Path(logFile.name)
    if debug and isNotebookRuntime():
        print(
            "GUNDAM native output is redirected to temporary log file "
            f"(auto-deleted after execution): {logPath}",
            flush=True,
        )

    try:
        with redirectNativeOutput(logPath, stream=stream):
            yield
    finally:
        try:
            if not stream and logPath.exists():
                logContent = logPath.read_text(encoding="utf-8", errors="replace")
                if logContent:
                    print(logContent, end="" if logContent.endswith("\n") else "\n")
        finally:
            logPath.unlink(missing_ok=True)


def _streamLogFile(
    path: Path,
    stopEvent: threading.Event,
    startOffset: int,
    outputFd: int,
    pollInterval: float = 0.1,
) -> None:
    """Print bytes appended to ``path`` until ``stopEvent`` is set."""
    try:
        with path.open("rb") as logFile:
            logFile.seek(startOffset)
            while not stopEvent.is_set():
                _streamAvailableLogBytes(logFile, outputFd)
                time.sleep(pollInterval)
            _streamAvailableLogBytes(logFile, outputFd)
    except OSError:
        return


def _streamAvailableLogBytes(logFile, outputFd: int) -> None:
    while True:
        chunk = logFile.read(8192)
        if not chunk:
            return
        if isNotebookRuntime():
            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            sys.stdout.flush()
        else:
            os.write(outputFd, chunk)
