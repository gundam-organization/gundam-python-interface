def test_configure_and_initialize_real_interface(configured_gundam_interface) -> None:
    assert configured_gundam_interface.engine is not None
