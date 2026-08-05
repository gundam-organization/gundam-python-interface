import numpy as np


def test_evaluate_likelihood(configured_gundam_interface) -> None:
    assert np.isfinite(configured_gundam_interface.evaluateLlh())
