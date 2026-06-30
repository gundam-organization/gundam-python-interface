from importlib.metadata import version

import gundam_interface


def test_package_exposes_version() -> None:
    assert gundam_interface.__version__ == version("gundam-interface")


def test_package_exposes_public_api() -> None:
    assert gundam_interface.GundamContext.__name__ == "GundamContext"
    assert gundam_interface.GundamInterface.__name__ == "GundamInterface"
    assert gundam_interface.GundamParameter.__name__ == "GundamParameter"
