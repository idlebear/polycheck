import numpy as np
import pytest

pytest.importorskip("pycuda")

try:
    import pycuda.driver as cuda

    cuda.init()
    if cuda.Device.count() == 0:
        pytest.skip("No CUDA device available", allow_module_level=True)
except Exception as exc:
    pytest.skip(f"CUDA unavailable: {exc}", allow_module_level=True)

import polycheck


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _expected_sensor_mask_grid(height, width, sensor):
    sx, sy, sensor_range, direction, fov = sensor
    sx = int(round(float(sx)))
    sy = int(round(float(sy)))

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dx = xx.astype(np.float32) - float(sx)
    dy = yy.astype(np.float32) - float(sy)

    if sensor_range <= 0.0:
        return (xx == sx) & (yy == sy)

    distance = np.sqrt(dx * dx + dy * dy)
    in_range = distance <= float(sensor_range) + 1e-6

    if 0.0 < float(fov) < (2.0 * np.pi - 1e-6):
        bearing = np.arctan2(dy, dx)
        angle_delta = np.abs(_wrap_to_pi(bearing - float(direction)))
        in_fov = (distance <= 1e-8) | (angle_delta <= (0.5 * float(fov) + 1e-6))
    else:
        in_fov = np.ones((height, width), dtype=bool)

    return in_range & in_fov


def _expected_sensor_mask_real(height, width, origin, resolution, sensor):
    sx, sy, sensor_range, direction, fov = sensor

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    tx = float(origin[0]) + (xx.astype(np.float32) + 0.5) * float(resolution)
    ty = float(origin[1]) + (yy.astype(np.float32) + 0.5) * float(resolution)

    if sensor_range <= 0.0:
        sensor_cell_x = int(np.floor((float(sx) - float(origin[0])) / float(resolution)))
        sensor_cell_y = int(np.floor((float(sy) - float(origin[1])) / float(resolution)))
        return (xx == sensor_cell_x) & (yy == sensor_cell_y)

    dx = tx - float(sx)
    dy = ty - float(sy)
    distance = np.sqrt(dx * dx + dy * dy)
    in_range = distance <= float(sensor_range) + 1e-6

    if 0.0 < float(fov) < (2.0 * np.pi - 1e-6):
        bearing = np.arctan2(dy, dx)
        angle_delta = np.abs(_wrap_to_pi(bearing - float(direction)))
        in_fov = (distance <= 1e-8) | (angle_delta <= (0.5 * float(fov) + 1e-6))
    else:
        in_fov = np.ones((height, width), dtype=bool)

    return in_range & in_fov


def test_sensor_zero_range_only_current_cell_visible_grid():
    data = np.full((7, 7), 0.25, dtype=np.float32)
    sensors = np.array([[3.0, 4.0, 0.0, 0.8, np.pi]], dtype=np.float32)

    per_sensor, union = polycheck.sensor_visibility_from_region(data, sensors)

    expected = np.zeros((7, 7), dtype=np.float32)
    expected[4, 3] = 1.0

    assert per_sensor.shape == (1, 7, 7)
    assert union.shape == (7, 7)
    assert np.allclose(per_sensor[0], expected, atol=1e-6)
    assert np.allclose(union, expected, atol=1e-6)


def test_sensor_zero_range_only_current_cell_visible_real():
    data = np.full((7, 7), 0.25, dtype=np.float32)
    origin = np.array([0.0, 0.0], dtype=np.float32)
    resolution = 1.0
    sensors = np.array([[3.5, 4.5, 0.0, 0.8, np.pi]], dtype=np.float32)

    per_sensor, union = polycheck.sensor_visibility_from_real_region(
        data, origin, resolution, sensors
    )

    expected = np.zeros((7, 7), dtype=np.float32)
    expected[4, 3] = 1.0

    assert per_sensor.shape == (1, 7, 7)
    assert union.shape == (7, 7)
    assert np.allclose(per_sensor[0], expected, atol=1e-6)
    assert np.allclose(union, expected, atol=1e-6)


@pytest.mark.parametrize(
    "sensor",
    [
        (4.0, 4.0, 1.0, 0.0, 2.0 * np.pi),
        (4.0, 4.0, 2.25, np.pi / 2.0, np.pi),
        (4.0, 4.0, 3.0, -np.pi / 4.0, np.pi / 2.0),
    ],
)
def test_sensor_range_and_fov_configurations_grid(sensor):
    data = np.zeros((9, 9), dtype=np.float32)
    sensors = np.array([sensor], dtype=np.float32)

    per_sensor, union = polycheck.sensor_visibility_from_region(data, sensors)
    expected_mask = _expected_sensor_mask_grid(9, 9, sensor).astype(np.float32)

    assert np.allclose(per_sensor[0], expected_mask, atol=1e-6)
    assert np.allclose(union, expected_mask, atol=1e-6)


@pytest.mark.parametrize(
    "sensor",
    [
        (4.5, 4.5, 1.0, 0.0, 2.0 * np.pi),
        (4.5, 4.5, 2.25, np.pi / 2.0, np.pi),
        (4.5, 4.5, 3.0, -np.pi / 4.0, np.pi / 2.0),
    ],
)
def test_sensor_range_and_fov_configurations_real(sensor):
    data = np.zeros((9, 9), dtype=np.float32)
    origin = np.array([0.0, 0.0], dtype=np.float32)
    resolution = 1.0
    sensors = np.array([sensor], dtype=np.float32)

    per_sensor, union = polycheck.sensor_visibility_from_real_region(
        data, origin, resolution, sensors
    )
    expected_mask = _expected_sensor_mask_real(
        9, 9, origin, resolution, sensor
    ).astype(np.float32)

    assert np.allclose(per_sensor[0], expected_mask, atol=1e-6)
    assert np.allclose(union, expected_mask, atol=1e-6)


def test_destination_cell_excluded_from_blocking_sum():
    sensors = np.array([[0.0, 2.0, 20.0, 0.0, np.pi / 2.0]], dtype=np.float32)

    data_without_target_occupancy = np.zeros((5, 5), dtype=np.float32)
    data_without_target_occupancy[2, 1] = 0.2
    data_without_target_occupancy[2, 2] = 0.3

    data_with_target_occupancy = data_without_target_occupancy.copy()
    data_with_target_occupancy[2, 4] = 1.0

    clear_no_target_occ, _ = polycheck.sensor_visibility_from_region(
        data_without_target_occupancy, sensors
    )
    clear_with_target_occ, _ = polycheck.sensor_visibility_from_region(
        data_with_target_occupancy, sensors
    )

    assert np.isclose(clear_no_target_occ[0, 2, 4], 0.5, atol=1e-6)
    assert np.isclose(clear_with_target_occ[0, 2, 4], 0.5, atol=1e-6)
    assert np.isclose(clear_no_target_occ[0, 2, 4], clear_with_target_occ[0, 2, 4])


def test_union_probability_matches_formula():
    data = np.zeros((6, 6), dtype=np.float32)
    data[3, 1] = 0.4  # blocker for sensor 0 ray to (3,3)
    data[1, 3] = 0.5  # blocker for sensor 1 ray to (3,3)

    sensors = np.array(
        [
            [0.0, 3.0, 20.0, 0.0, 2.0 * np.pi],
            [3.0, 0.0, 20.0, np.pi / 2.0, 2.0 * np.pi],
        ],
        dtype=np.float32,
    )

    per_sensor, union = polycheck.sensor_visibility_from_region(data, sensors)

    assert np.isclose(per_sensor[0, 3, 3], 0.6, atol=1e-6)
    assert np.isclose(per_sensor[1, 3, 3], 0.5, atol=1e-6)
    assert np.isclose(union[3, 3], 1.0 - (1.0 - 0.6) * (1.0 - 0.5), atol=1e-6)

    expected_union = 1.0 - np.prod(1.0 - per_sensor, axis=0)
    assert np.allclose(union, expected_union, atol=1e-6)


def test_product_combination_ignores_out_of_coverage_sensors():
    data = np.zeros((7, 7), dtype=np.float32)
    data[3, 4] = 0.3  # blocker on sensor 0 ray to (3,5)

    sensors = np.array(
        [
            [3.0, 3.0, 5.0, 0.0, np.pi / 2.0],  # covers cell (3,5)
            [0.0, 0.0, 1.0, 0.0, np.pi / 4.0],  # does not cover cell (3,5)
        ],
        dtype=np.float32,
    )

    per_sensor, combined_product = polycheck.sensor_visibility_from_region(
        data, sensors, combine="product"
    )

    # For (3,5): only sensor 0 contributes. Sensor 1 is out-of-coverage and must be neutral.
    assert np.isclose(per_sensor[0, 3, 5], 0.7, atol=1e-6)
    assert np.isclose(per_sensor[1, 3, 5], 0.0, atol=1e-6)
    assert np.isclose(combined_product[3, 5], 0.7, atol=1e-6)
