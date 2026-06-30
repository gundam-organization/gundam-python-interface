import gundam_interface


def test_package_exposes_version() -> None:
    assert gundam_interface.__version__ == "0.1.0"


def test_package_exposes_public_api() -> None:
    assert gundam_interface.GundamContext.__name__ == "GundamContext"
    assert gundam_interface.GundamInterface.__name__ == "GundamInterface"
    assert gundam_interface.GundamParameter.__name__ == "GundamParameter"
