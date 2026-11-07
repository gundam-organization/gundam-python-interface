import sys
from types import SimpleNamespace

import pytest

from gundam_interface.internal.minimizer import GundamMinimizer


class RootMinimizer:
    def isRootMinimizer(self):
        return True

    def getFunctionPrecision(self):
        return 1e-8

    def hasConverged(self):
        return False

    def isMinimizeDone(self):
        return True


class DerivedRootMinimizer(RootMinimizer):
    pass


@pytest.fixture(autouse=True)
def no_gundam_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "GUNDAM", None)


@pytest.mark.parametrize("handle_type", [RootMinimizer, DerivedRootMinimizer])
def test_root_minimizer_methods_accept_subclasses(handle_type):
    minimizer = GundamMinimizer(handle_type())

    assert minimizer.isRootMinimizer()
    assert minimizer.getFunctionPrecision() == 1e-8
    assert minimizer.hasConverged() is False
    assert minimizer.isMinimizeDone() is True


@pytest.mark.parametrize("method", ["getFunctionPrecision", "hasConverged", "isMinimizeDone"])
def test_root_minimizer_methods_reject_unrelated_handles(method):
    handle = SimpleNamespace(
        isRootMinimizer=lambda: False,
        **{method: lambda: pytest.fail("Unexpected delegation")},
    )
    minimizer = GundamMinimizer(handle)

    assert not minimizer.isRootMinimizer()
    with pytest.raises(TypeError, match="requires a RootMinimizer"):
        getattr(minimizer, method)()


def test_base_status_does_not_require_root_bindings(monkeypatch):
    monkeypatch.setitem(sys.modules, "GUNDAM", None)
    minimizer = GundamMinimizer(SimpleNamespace(getMinimizerStatus=lambda: 0))

    assert minimizer.getMinimizerStatus() == 0
