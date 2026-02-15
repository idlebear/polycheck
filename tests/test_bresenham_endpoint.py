import numpy as np
import pytest


@pytest.mark.parametrize("module_name", ["polycheck.poly_warp", "polycheck"])
def test_destination_cell_blocks_visibility(module_name):
    """The destination cell occupancy should zero out visibility."""
    module = pytest.importorskip(module_name)

    data = np.zeros((3, 3), dtype=np.float32)
    data[1, 2] = 1.0  # Block the target cell

    start = np.array([0, 1], dtype=np.int32)
    ends = np.array([[2, 1]], dtype=np.int32)

    visibility = module.visibility(data, start, ends)
    assert np.isclose(visibility[1, 2], 0.0, atol=1e-6)


@pytest.mark.parametrize("module_name", ["polycheck.poly_warp", "polycheck"])
def test_destination_cell_counted_once(module_name):
    """Endpoint occupancy should be applied exactly once."""
    module = pytest.importorskip(module_name)

    data = np.zeros((3, 3), dtype=np.float32)
    data[1, 2] = 0.5  # Semi-occupied target cell

    start = np.array([0, 1], dtype=np.int32)
    ends = np.array([[2, 1]], dtype=np.int32)

    visibility = module.visibility(data, start, ends)
    assert np.isclose(visibility[1, 2], 0.5, atol=1e-6)
