def test_access_model_and_data_samples(configured_gundam_interface) -> None:
    model = configured_gundam_interface.getModel()
    data = configured_gundam_interface.getData()

    modelSample = model.getSampleSet().getSampleList()[0]
    dataSample = data.getSampleSet().getSampleList()[0]
    assert len(model.getSampleSet().getSampleList()) == 1
    assert len(data.getSampleSet().getSampleList()) == 1
    assert [bin.getSumWeights() for bin in modelSample.getHistogram().getBinList()] == [
        10.0,
        0.0,
    ]
    assert [bin.getSumWeights() for bin in dataSample.getHistogram().getBinList()] == [
        10.0,
        0.0,
    ]
