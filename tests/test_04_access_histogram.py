def test_access_histogram(configured_gundam_interface) -> None:
    sample = configured_gundam_interface.getModel().getSampleSet().getSampleList()[0]
    histogram = sample.getHistogram()

    assert histogram.getNbBins() == 2
    assert len(histogram.getBinList()) == 2
