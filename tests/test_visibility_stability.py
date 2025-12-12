import pytest
import numpy as np

# Skip the test if warp is missing, but allow it to run in CI/local envs where it's present.
pytest.importorskip("warp")

import polycheck.poly_warp as pw


def check_visibility_smoothness(visibility_results, path_type="grid"):
    """
    Analyzes visibility results for smoothness.

    Args:
        visibility_results (np.ndarray): Array of visibility values.
        path_type (str): Identifier for the test type ('grid' or 'real').

    Returns:
        bool: True if the results are smooth, False otherwise.
    """
    # The path is designed to have a clear view, then be occluded, then clear again.
    # We expect to see a sequence of 1s, then 0s, then 1s.

    # Find the first and last points where visibility is occluded (close to 0).
    occluded_indices = np.where(visibility_results < 0.1)[0]

    if len(occluded_indices) == 0:
        print(
            f"✗ [{path_type.upper()}] FAILED: Path was never occluded, check test setup."
        )
        return False

    first_occluded = occluded_indices[0]
    last_occluded = occluded_indices[-1]

    # 1. Check the "clear view" section before occlusion.
    # All values should be close to 1.0.
    pre_occlusion_slice = visibility_results[:first_occluded]
    if not np.all(pre_occlusion_slice > 0.99):
        bad_indices = np.where(pre_occlusion_slice <= 0.99)[0]
        print(
            f"✗ [{path_type.upper()}] FAILED: Visibility spikes/dips found before occlusion at indices: {bad_indices}"
        )
        print(f"   Values: {visibility_results[bad_indices]}")
        return False

    # 2. Check the "occluded" section.
    # All values should be close to 0.0.
    occlusion_slice = visibility_results[first_occluded : last_occluded + 1]
    if not np.all(occlusion_slice < 0.01):
        bad_indices = np.where(occlusion_slice >= 0.01)[0] + first_occluded
        print(
            f"✗ [{path_type.upper()}] FAILED: Unexpected visibility during occlusion at indices: {bad_indices}"
        )
        print(f"   Values: {visibility_results[bad_indices]}")
        return False

    # 3. Check the "clear view" section after occlusion.
    # All values should be close to 1.0.
    post_occlusion_slice = visibility_results[last_occluded + 1 :]
    if not np.all(post_occlusion_slice > 0.99):
        bad_indices = np.where(post_occlusion_slice <= 0.99)[0] + last_occluded + 1
        print(
            f"✗ [{path_type.upper()}] FAILED: Visibility spikes/dips found after occlusion at indices: {bad_indices}"
        )
        print(f"   Values: {visibility_results[bad_indices]}")
        return False

    print(f"✓ [{path_type.upper()}] PASSED: Visibility varies smoothly as expected.")
    return True


def test_visibility_from_region_stability():
    """
    Tests the stability of `visibility_from_region` by moving an observer
    along a path and checking for smooth transitions in visibility.
    """
    print("\n--- Testing visibility_from_region() stability ---")
    # 1. Setup: A grid with a solid obstacle in the middle.
    grid_size = 30
    obstacle_size = 4
    obstacle_start = (grid_size - obstacle_size) // 2
    obstacle_end = obstacle_start + obstacle_size

    data = np.zeros((grid_size, grid_size), dtype=np.float32)
    data[obstacle_start:obstacle_end, obstacle_start:obstacle_end] = 1.0

    # 2. Path: A horizontal line of points for the observer.
    # This path starts with a clear view, goes behind the obstacle, and emerges.
    path_y = grid_size // 2
    path_points = np.array(
        [[x, path_y] for x in range(1, grid_size - 1)], dtype=np.int32
    )

    # 3. Target: A single point to observe.
    target_point = np.array([[grid_size // 2, 1]], dtype=np.int32)

    # 4. Execution: Calculate visibility from all path points to the target.
    visibility_results = pw.visibility_from_region(data, path_points, target_point)

    # 5. Analysis: Check for the expected smooth 1 -> 0 -> 1 transition.
    assert check_visibility_smoothness(visibility_results, "grid")


def test_visibility_from_real_region_stability():
    """
    Tests the stability of `visibility_from_real_region` by moving an observer
    along a path and checking for smooth transitions in visibility.
    """
    print("\n--- Testing visibility_from_real_region() stability ---")
    # 1. Setup: A grid with a solid obstacle in the middle.
    grid_size = 30
    obstacle_size = 4
    obstacle_start = (grid_size - obstacle_size) // 2
    obstacle_end = obstacle_start + obstacle_size

    data = np.zeros((grid_size, grid_size), dtype=np.float32)
    data[obstacle_start:obstacle_end, obstacle_start:obstacle_end] = 1.0

    origin = np.array([0.0, 0.0], dtype=np.float32)
    resolution = 0.1  # 10cm resolution

    # 2. Path: A horizontal line of points for the observer in real-world coordinates.
    # The path is sampled at a finer resolution than the grid itself.
    path_y_real = (grid_size / 2) * resolution
    path_x_coords = np.linspace(
        1.5 * resolution, (grid_size - 1.5) * resolution, num=200
    )
    path_points = np.array([[x, path_y_real] for x in path_x_coords], dtype=np.float32)

    # 3. Target: A single point to observe in real-world coordinates.
    target_x_real = (grid_size / 2) * resolution
    target_y_real = 1.5 * resolution
    target_point = np.array([[target_x_real, target_y_real]], dtype=np.float32)

    # 4. Execution: Calculate visibility from all path points to the target.
    visibility_results = pw.visibility_from_real_region(
        data, origin, resolution, path_points, target_point
    )

    # 5. Analysis: Check for the expected smooth 1 -> 0 -> 1 transition.
    assert check_visibility_smoothness(visibility_results, "real")
