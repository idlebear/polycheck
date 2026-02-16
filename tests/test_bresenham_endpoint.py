import numpy as np
import pytest


@pytest.mark.parametrize("module_name", ["polycheck.poly_warp", "polycheck"])
def test_destination_cell_is_excluded_from_visibility(module_name):
    """Endpoint occupancy should not affect visibility."""
    module = pytest.importorskip(module_name)

    data = np.zeros((3, 3), dtype=np.float32)
    data[1, 2] = 1.0  # Block the target cell

    start = np.array([0, 1], dtype=np.int32)
    ends = np.array([[2, 1]], dtype=np.int32)

    visibility = module.visibility(data, start, ends)
    assert np.isclose(visibility[1, 2], 1.0, atol=1e-6)


@pytest.mark.parametrize("module_name", ["polycheck.poly_warp", "polycheck"])
def test_destination_cell_occupancy_ignored_for_fractional_values(module_name):
    """Endpoint occupancy should be ignored, even for fractional occupancy."""
    module = pytest.importorskip(module_name)

    data = np.zeros((3, 3), dtype=np.float32)
    data[1, 2] = 0.5  # Semi-occupied target cell

    start = np.array([0, 1], dtype=np.int32)
    ends = np.array([[2, 1]], dtype=np.int32)

    visibility = module.visibility(data, start, ends)
    assert np.isclose(visibility[1, 2], 1.0, atol=1e-6)
