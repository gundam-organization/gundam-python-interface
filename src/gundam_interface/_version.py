"""Package version."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gundam-interface")
except PackageNotFoundError:
    __version__ = "0+unknown"
