import uproot


def test_generate_gundam_inputs(gundam_test_inputs) -> None:
    assert gundam_test_inputs.rootPath.is_file()
    assert gundam_test_inputs.configPath.is_file()

    with uproot.open(gundam_test_inputs.rootPath) as rootFile:
        assert rootFile["tree_mc"].num_entries == 10
    assert "fitterEngineConfig:" in gundam_test_inputs.configPath.read_text(
        encoding="utf-8"
    )
