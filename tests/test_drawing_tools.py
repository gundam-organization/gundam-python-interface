import numpy as np

from gundam_interface.drawing_tools import _computeDensity


def test_density_counts_each_dial_once_per_crossed_bin():
    values = np.array([0.0, 0.25, 0.75, 1.0])
    # The first dial crosses both y bins in each x bin. The second stays low.
    responses = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

    xEdges, yEdges, density = _computeDensity(values, responses, nBins=2)

    np.testing.assert_array_equal(xEdges, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(yEdges, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(density, [[100.0, 50.0], [100.0, 50.0]])
