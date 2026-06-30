from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
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


@contextmanager
def redirectNativeOutput(
    logPath: str | os.PathLike[str],
    *,
    stream: bool = False,
) -> Iterator[None]:
    """Redirect C/C++ stdout and stderr to a file.

    This is useful in Jupyter where native C++ loggers can interact poorly with
    ipykernel stdout capture. Python stdout/stderr are restored after the block.
    When ``stream`` is true, new log content is also printed as it is written.
    """
    path = Path(logPath).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

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
):
    """Redirect native output to a user path or to an auto-deleted temporary file."""
    if logPath is None:
        return temporaryRedirectNativeOutput(prefix=prefix, stream=stream)
    return redirectNativeOutput(logPath, stream=stream)


@contextmanager
def temporaryRedirectNativeOutput(
    prefix: str = "gundam",
    *,
    stream: bool = False,
) -> Iterator[None]:
    """Redirect native output to a temporary log file and delete it afterwards."""
    with tempfile.NamedTemporaryFile(prefix=f"{prefix}_", suffix=".log", delete=False) as logFile:
        logPath = Path(logFile.name)

    try:
        with redirectNativeOutput(logPath, stream=stream):
            yield
    finally:
        try:
            if isNotebookRuntime() and logPath.exists():
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
