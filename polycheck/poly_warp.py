"""
This module provides Warp-based implementations of geometric and visibility check functions,
porting the PyCUDA functions from polycheck.py to use NVIDIA Warp for GPU acceleration.

Functions:
    contains(poly, points):
        Checks if a set of points are inside a given polygon using the winding number algorithm.

    visibility(data, start, ends, max_range=None):
        Computes visibility from a start point to multiple end points on a grid using Bresenham's line algorithm.

    visibility_from_region(data, starts, ends, max_range=None):
        Computes visibility from multiple start points to multiple end points on a grid using Bresenham's line algorithm.

    visibility_from_real_region(data, origin, resolution, starts, ends, max_range=None):
        Computes visibility from multiple start points to multiple end points on a grid with real-world coordinates.

    faux_scan(polygons, origin, angle_start, angle_inc, num_rays, max_range, resolution):
        Performs a faux laser scan from an origin point, simulating rays at specified angles.
"""

import warp as wp
import numpy as np

# Initialize Warp with quiet mode to suppress status messages
wp.config.quiet = True
wp.init()


# Utility functions
@wp.func
def is_zero(f: float) -> bool:
    """Check if a float is effectively zero within epsilon tolerance"""
    return f >= -2e-6 and f <= 2e-6


@wp.func
def side(v1x: float, v1y: float, v2x: float, v2y: float, px: float, py: float) -> float:
    """
    Check the position of a point relative to a directional infinite line.
    Returns: +ve if point is left of v1->v2, 0 if on the line, -ve otherwise
    """
    return (v2x - v1x) * (py - v1y) - (px - v1x) * (v2y - v1y)


@wp.func
def test_point_in_polygon(
    polygon_data: wp.array(dtype=float), num_vertices: int, px: float, py: float
) -> bool:
    """Test if a point is inside a polygon using winding number algorithm"""
    winding_number = int(0)  # Dynamic variable
    for vertex in range(num_vertices):
        v1x = polygon_data[2 * vertex]
        v1y = polygon_data[2 * vertex + 1]
        next_vertex = (vertex + 1) % num_vertices
        v2x = polygon_data[2 * next_vertex]
        v2y = polygon_data[2 * next_vertex + 1]

        if v1y <= py:
            if v2y > py:
                if side(v1x, v1y, v2x, v2y, px, py) > 0.0:
                    winding_number += 1
        else:
            if v2y <= py:
                if side(v1x, v1y, v2x, v2y, px, py) < 0.0:
                    winding_number -= 1

    return winding_number != 0


@wp.kernel
def check_points_kernel(
    polygon_data: wp.array(dtype=float),
    num_vertices: int,
    points_data: wp.array(dtype=float),
    num_points: int,
    results: wp.array(dtype=float),
):
    """Kernel to check if points are inside polygon"""
    i = wp.tid()
    if i < num_points:
        px = points_data[2 * i]
        py = points_data[2 * i + 1]
        results[i] = float(test_point_in_polygon(polygon_data, num_vertices, px, py))


@wp.func
def line_observation_bresenham(
    data: wp.array(dtype=float),
    height: int,
    width: int,
    sx: int,
    sy: int,
    ex: int,
    ey: int,
    max_range: int,
) -> float:
    """Compute line observation using Bresenham's line algorithm"""
    dx = wp.abs(sx - ex)
    step_x = 1 if sx < ex else -1
    dy = -wp.abs(sy - ey)
    step_y = 1 if sy < ey else -1
    error = dx + dy
    steps = int(0)  # Dynamic variable

    if sx == ex and sy == ey:
        return 1.0

    observation = float(1.0)  # Dynamic variable
    current_x = int(sx)  # Dynamic variable
    current_y = int(sy)  # Dynamic variable
    continue_loop = int(1)  # Dynamic variable

    while continue_loop == 1:
        e2 = 2 * error
        if e2 >= dy:
            if current_x == ex:
                continue_loop = 0
            else:
                error += dy
                current_x += step_x
        if e2 <= dx and continue_loop == 1:
            if current_y == ey:
                continue_loop = 0
            else:
                error += dx
                current_y += step_y

        if continue_loop == 1 and current_x == ex and current_y == ey:
            continue_loop = 0

        if continue_loop == 1:
            steps += 1
            if max_range > 0 and steps > max_range:
                observation = 0.0
                continue_loop = 0

            # Check bounds
            if (
                current_x < 0
                or current_x >= width
                or current_y < 0
                or current_y >= height
            ):
                observation = 0.0
                continue_loop = 0

            # Apply view probability
            if continue_loop == 1:
                observation *= 1.0 - data[current_y * width + current_x]
                if observation < 2e-6:
                    observation = 0.0
                    continue_loop = 0

    return observation


@wp.func
def line_observation_real(
    data: wp.array(dtype=float),
    height: int,
    width: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
    src_x: float,
    src_y: float,
    end_x: float,
    end_y: float,
    max_range: float,
) -> float:
    """Compute line observation using floating point grid traversal"""
    dx = end_x - src_x
    dy = end_y - src_y
    magnitude = wp.sqrt(dx * dx + dy * dy)

    # If the target is beyond max_range, it's not visible.
    if max_range > 0.0 and magnitude > max_range:
        return 0.0

    # Handle zero magnitude case
    if is_zero(magnitude):
        s_check_x = int(wp.floor((src_x - origin_x) / resolution))
        s_check_y = int(wp.floor((src_y - origin_y) / resolution))
        if s_check_x < 0 or s_check_x >= width or s_check_y < 0 or s_check_y >= height:
            return 0.0
        return 1.0

    dx /= magnitude
    dy /= magnitude

    rx = (src_x - origin_x) / resolution
    sx = int(wp.floor(rx))
    ry = (src_y - origin_y) / resolution
    sy = int(wp.floor(ry))

    # Check if starting point is outside grid
    if sx < 0 or sx >= width or sy < 0 or sy >= height:
        return 0.0

    ex = int(wp.floor((end_x - origin_x) / resolution))
    ey = int(wp.floor((end_y - origin_y) / resolution))

    if sx == ex and sy == ey:
        return 1.0

    # Calculate the distance to the center of the target cell for a robust termination check
    end_center_x = (float(ex) + 0.5) * resolution + origin_x
    end_center_y = (float(ey) + 0.5) * resolution + origin_y
    dist_to_end_center = wp.sqrt(
        (end_center_x - src_x) * (end_center_x - src_x)
        + (end_center_y - src_y) * (end_center_y - src_y)
    )

    # Set up grid traversal
    step_x = 0
    t_max_x = float(0.0)  # Dynamic variable
    t_delta_x = float(0.0)  # Dynamic variable

    if is_zero(dx):
        step_x = 0
        t_max_x = 1e30
        t_delta_x = 1e30
    elif dx > 0.0:
        step_x = 1
        t_max_x = (wp.floor(rx) + 1.0 - rx) * resolution / dx
        t_delta_x = resolution / dx
    else:
        step_x = -1
        t_max_x = (rx - wp.floor(rx)) * resolution / (-dx)
        t_delta_x = resolution / (-dx)

    step_y = 0
    t_max_y = float(0.0)  # Dynamic variable
    t_delta_y = float(0.0)  # Dynamic variable

    if is_zero(dy):
        step_y = 0
        t_max_y = 1e30
        t_delta_y = 1e30
    elif dy > 0.0:
        step_y = 1
        t_max_y = (wp.floor(ry) + 1.0 - ry) * resolution / dy
        t_delta_y = resolution / dy
    else:
        step_y = -1
        t_max_y = (ry - wp.floor(ry)) * resolution / (-dy)
        t_delta_y = resolution / (-dy)

    observation = float(1.0)  # Dynamic variable
    current_sx = int(sx)  # Dynamic variable
    current_sy = int(sy)  # Dynamic variable

    while True:
        # Robust termination check: stop if we've passed the center of the target cell
        if wp.min(t_max_x, t_max_y) > magnitude:  # dist_to_end_center:
            break

        if t_max_x < t_max_y:
            current_sx += step_x
            t_max_x += t_delta_x
        else:
            current_sy += step_y
            t_max_y += t_delta_y

        # Check if reached target
        if current_sx == ex and current_sy == ey:
            break

        # Check grid boundaries
        if (
            current_sx < 0
            or current_sx >= width
            or current_sy < 0
            or current_sy >= height
        ):
            observation = 0.0
            break

        # Apply view probability
        observation *= 1.0 - data[current_sy * width + current_sx]
        if is_zero(observation):
            observation = 0.0
            break

    return observation


@wp.kernel
def check_visibility_kernel(
    data: wp.array(dtype=float),
    height: int,
    width: int,
    start: wp.array(dtype=int),
    ends: wp.array(dtype=int),
    num_ends: int,
    max_range: int,
    results: wp.array(dtype=float),
):
    """Kernel for visibility computation from single start point"""
    i = wp.tid()
    if i < num_ends:
        ex = ends[2 * i]
        ey = ends[2 * i + 1]
        results[ey * width + ex] = line_observation_bresenham(
            data, height, width, start[0], start[1], ex, ey, max_range
        )


@wp.kernel
def check_region_visibility_kernel(
    data: wp.array(dtype=float),
    height: int,
    width: int,
    starts: wp.array(dtype=int),
    num_starts: int,
    ends: wp.array(dtype=int),
    num_ends: int,
    max_range: int,
    results: wp.array(dtype=float),
):
    """Kernel for visibility computation from multiple start points to multiple end points"""
    tid = wp.tid()

    # Calculate 2D indices from 1D thread ID
    ends_stride = (num_ends + 31) // 32 * 32  # Round up to multiple of 32
    si = tid // ends_stride
    ei = tid % ends_stride

    if si < num_starts and ei < num_ends:
        sx = starts[2 * si]
        sy = starts[2 * si + 1]
        ex = ends[2 * ei]
        ey = ends[2 * ei + 1]
        results[si * num_ends + ei] = line_observation_bresenham(
            data, height, width, sx, sy, ex, ey, max_range
        )


@wp.kernel
def check_real_region_visibility_kernel(
    data: wp.array(dtype=float),
    height: int,
    width: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
    starts: wp.array(dtype=float),
    num_starts: int,
    ends: wp.array(dtype=float),
    num_ends: int,
    max_range: float,
    results: wp.array(dtype=float),
):
    """Kernel for real-coordinate visibility computation from multiple start points"""
    tid = wp.tid()

    # Calculate 2D indices from 1D thread ID
    ends_stride = (num_ends + 31) // 32 * 32  # Round up to multiple of 32
    si = tid // ends_stride
    ei = tid % ends_stride

    if si < num_starts and ei < num_ends:
        sx = starts[2 * si]
        sy = starts[2 * si + 1]
        ex = ends[2 * ei]
        ey = ends[2 * ei + 1]
        results[si * num_ends + ei] = line_observation_real(
            data,
            height,
            width,
            origin_x,
            origin_y,
            resolution,
            sx,
            sy,
            ex,
            ey,
            max_range,
        )


@wp.func
def test_point_in_polygon_indexed(
    polygon_data: wp.array(dtype=float),
    start_idx: int,
    num_vertices: int,
    px: float,
    py: float,
) -> bool:
    """Test if a point is inside a polygon with indexed access"""
    winding_number = int(0)  # Dynamic variable
    for vertex in range(num_vertices):
        # Calculate flat array indices for 2D coordinates
        vertex_idx = start_idx + vertex
        v1x = polygon_data[2 * vertex_idx]
        v1y = polygon_data[2 * vertex_idx + 1]
        next_vertex = (vertex + 1) % num_vertices
        next_vertex_idx = start_idx + next_vertex
        v2x = polygon_data[2 * next_vertex_idx]
        v2y = polygon_data[2 * next_vertex_idx + 1]

        if v1y <= py:
            if v2y > py:
                if side(v1x, v1y, v2x, v2y, px, py) > 0.0:
                    winding_number += 1
        else:
            if v2y <= py:
                if side(v1x, v1y, v2x, v2y, px, py) < 0.0:
                    winding_number -= 1

    return winding_number != 0


@wp.func
def line_range(
    polygon_list: wp.array(dtype=wp.float32),
    polygon_indices: wp.array(dtype=wp.int32),
    num_polygons: int,
    sx: float,
    sy: float,
    angle: float,
    max_range: float,
    resolution: float,
):
    """Cast a ray and find range to first polygon intersection"""
    ex = float(sx)  # Dynamic variable
    ey = float(sy)  # Dynamic variable
    x_inc = wp.cos(angle) * resolution
    y_inc = wp.sin(angle) * resolution
    dist = float(0.0)  # Dynamic variable

    while dist < max_range:
        ex += x_inc
        ey += y_inc

        for i in range(num_polygons):
            start_idx = polygon_indices[i]
            end_idx = polygon_indices[i + 1]
            num_vertices = end_idx - start_idx

            if test_point_in_polygon_indexed(
                polygon_list, start_idx, num_vertices, ex, ey
            ):
                return dist, i  # Return distance and polygon index

        dist += resolution

    return -1.0, -1


@wp.kernel
def faux_ray_kernel(
    polygon_list: wp.array(dtype=wp.float32),
    polygon_indices: wp.array(dtype=wp.int32),
    num_polygons: int,
    start_x: float,
    start_y: float,
    angle_start: float,
    angle_increment: float,
    num_rays: int,
    max_range: float,
    resolution: float,
    results: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
):
    """Kernel for faux laser scan computation"""
    i = wp.tid()
    if i < num_rays:
        angle = angle_start + float(i) * angle_increment
        result, index = line_range(
            polygon_list,
            polygon_indices,
            num_polygons,
            start_x,
            start_y,
            angle,
            max_range,
            resolution,
        )
        results[i] = result
        indices[i] = index


def contains(poly, points):
    """
    Check if a set of points are inside a given polygon using the winding number algorithm.

    Args:
        poly: numpy array of polygon vertices, shape (n_vertices, 2)
        points: numpy array of points to test, shape (n_points, 2)

    Returns:
        numpy array of boolean results, shape (n_points,)
    """
    poly = np.asarray(poly, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)

    num_vertices = len(poly)
    num_points = len(points)

    # Ensure 2D arrays
    if poly.ndim == 1:
        poly = poly.reshape(-1, 2)
    if points.ndim == 1:
        points = points.reshape(-1, 2)

    # Flatten arrays for Warp kernels
    poly_flat = poly.flatten()
    points_flat = points.flatten()

    # Create Warp arrays
    poly_wp = wp.array(poly_flat, dtype=wp.float32, device="cuda")
    points_wp = wp.array(points_flat, dtype=wp.float32, device="cuda")
    results_wp = wp.zeros(num_points, dtype=wp.float32, device="cuda")

    # Launch kernel
    wp.launch(
        check_points_kernel,
        dim=num_points,
        inputs=[poly_wp, num_vertices, points_wp, num_points, results_wp],
    )

    # Return results as numpy array
    return results_wp.numpy()


def visibility(data, start, ends, max_range=None):
    """
    Compute visibility from a start point to multiple end points on a grid using Bresenham's line algorithm.

    Args:
        data: 2D numpy array representing the occupancy grid
        start: numpy array [x, y] start coordinates (grid indices)
        ends: numpy array of end points, shape (n_points, 2) (grid indices)
        max_range: maximum range to check (in grid cells), None for unlimited

    Returns:
        2D numpy array same shape as data with visibility values
    """
    data = np.asarray(data, dtype=np.float32)
    start = np.asarray(start, dtype=np.int32)
    ends = np.asarray(ends, dtype=np.int32)

    height, width = data.shape
    num_ends = len(ends)

    if max_range is None:
        max_range = 0
    else:
        max_range = int(max_range)

    # Ensure 2D arrays
    if ends.ndim == 1:
        ends = ends.reshape(-1, 2)

    # Flatten arrays for Warp kernels
    ends_flat = ends.flatten()
    data_flat = data.flatten()

    # Create Warp arrays
    data_wp = wp.array(data_flat, dtype=wp.float32, device="cuda")
    start_wp = wp.array(start, dtype=wp.int32, device="cuda")
    ends_wp = wp.array(ends_flat, dtype=wp.int32, device="cuda")
    results_wp = wp.zeros(height * width, dtype=wp.float32, device="cuda")

    # Launch kernel
    wp.launch(
        check_visibility_kernel,
        dim=num_ends,
        inputs=[
            data_wp,
            height,
            width,
            start_wp,
            ends_wp,
            num_ends,
            max_range,
            results_wp,
        ],
    )

    return results_wp.numpy().reshape(height, width)


def visibility_from_region(data, starts, ends, max_range=None):
    """
    Compute visibility from multiple start points to multiple end points on a grid using Bresenham's line algorithm.

    Args:
        data: 2D numpy array representing the occupancy grid
        starts: numpy array of start points, shape (n_starts, 2) (grid indices)
        ends: numpy array of end points, shape (n_ends, 2) (grid indices)
        max_range: maximum range to check (in grid cells), None for unlimited

    Returns:
        1D numpy array of visibility values, shape (n_starts * n_ends)
    """
    data = np.asarray(data, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.int32)
    ends = np.asarray(ends, dtype=np.int32)

    height, width = data.shape
    num_starts = len(starts)
    num_ends = len(ends)

    if max_range is None:
        max_range = 0
    else:
        max_range = int(max_range)

    # Ensure 2D arrays
    if starts.ndim == 1:
        starts = starts.reshape(-1, 2)
    if ends.ndim == 1:
        ends = ends.reshape(-1, 2)

    # Flatten arrays for Warp kernels
    starts_flat = starts.flatten()
    ends_flat = ends.flatten()
    data_flat = data.flatten()

    # Create Warp arrays
    data_wp = wp.array(data_flat, dtype=wp.float32, device="cuda")
    starts_wp = wp.array(starts_flat, dtype=wp.int32, device="cuda")
    ends_wp = wp.array(ends_flat, dtype=wp.int32, device="cuda")
    results_wp = wp.zeros(num_starts * num_ends, dtype=wp.float32, device="cuda")

    # Launch kernel with enough threads for all start-end combinations
    total_threads = num_starts * (
        (num_ends + 31) // 32 * 32
    )  # Round up ends to multiple of 32
    wp.launch(
        check_region_visibility_kernel,
        dim=total_threads,
        inputs=[
            data_wp,
            height,
            width,
            starts_wp,
            num_starts,
            ends_wp,
            num_ends,
            max_range,
            results_wp,
        ],
    )

    return results_wp.numpy()


def visibility_from_real_region(data, origin, resolution, starts, ends, max_range=None):
    """
    Compute visibility from multiple start points to multiple end points on a grid with real-world coordinates.

    Args:
        data: 2D numpy array representing the occupancy grid
        origin: [x, y] origin coordinates in real-world units
        resolution: grid resolution in real-world units per cell
        starts: numpy array of start points in real-world coordinates, shape (n_starts, 2)
        ends: numpy array of end points in real-world coordinates, shape (n_ends, 2)
        max_range: maximum range to check in real-world units, None for unlimited

    Returns:
        1D numpy array of visibility values, shape (n_starts * n_ends)
    """

    # data = np.asarray(data, dtype=np.float32)
    # origin = np.asarray(origin, dtype=np.float32)
    # starts = np.asarray(starts, dtype=np.float32)
    # ends = np.asarray(ends, dtype=np.float32)

    height, width = data.shape
    num_starts = len(starts)
    num_ends = len(ends)

    if max_range is None:
        max_range = 0.0
    else:
        max_range = float(max_range)

    assert (
        starts.ndim == 2 and starts.shape[1] == 2
    ), "Starts must be a 2D array with shape (n, 2)"
    assert (
        ends.ndim == 2 and ends.shape[1] == 2
    ), "Ends must be a 2D array with shape (n, 2)"

    # Flatten arrays for Warp kernels
    starts_flat = starts.ravel()
    ends_flat = ends.ravel()
    data_flat = data.ravel()

    # Create Warp arrays
    data_wp = wp.array(data_flat, dtype=wp.float32, device="cuda")
    starts_wp = wp.array(starts_flat, dtype=wp.float32, device="cuda")
    ends_wp = wp.array(ends_flat, dtype=wp.float32, device="cuda")
    results_wp = wp.zeros(num_starts * num_ends, dtype=wp.float32, device="cuda")

    # Launch kernel with enough threads for all start-end combinations
    total_threads = num_starts * (
        (num_ends + 31) // 32 * 32
    )  # Round up ends to multiple of 32
    wp.launch(
        check_real_region_visibility_kernel,
        dim=total_threads,
        inputs=[
            data_wp,
            height,
            width,
            origin[0],
            origin[1],
            float(resolution),
            starts_wp,
            num_starts,
            ends_wp,
            num_ends,
            max_range,
            results_wp,
        ],
    )

    return results_wp.numpy()


def faux_scan(
    polygons, origin, angle_start, angle_inc, num_rays, max_range, resolution
):
    """
    Perform a faux laser scan from an origin point, simulating rays at specified angles and increments.

    Args:
        polygons: list of numpy arrays, each representing a polygon's vertices
        origin: [x, y] origin point for the scan
        angle_start: starting angle in radians
        angle_inc: angle increment between rays in radians
        num_rays: number of rays to cast
        max_range: maximum range for rays in real-world units
        resolution: resolution for ray stepping in real-world units

    Returns:
        numpy array of ranges to first obstacle for each ray, -1.0 if no obstacle found
    """
    if polygons is None or len(polygons) == 0:
        return (
            np.ones((num_rays,), dtype=np.float32) * -1.0,
            np.ones((num_rays,), dtype=np.int32) * 0x7FFFFFFF,
        )

    # Flatten all polygons into a single array and create indices
    all_vertices = []
    polygon_indices = [0]

    for poly in polygons:
        poly = np.asarray(poly, dtype=np.float32)
        if poly.ndim == 1:
            poly = poly.reshape(-1, 2)
        all_vertices.extend(poly.flatten())
        polygon_indices.append(
            len(all_vertices) // 2
        )  # Number of vertices, not coordinates

    all_vertices = np.array(all_vertices, dtype=np.float32)
    polygon_indices = np.array(polygon_indices, dtype=np.int32)

    # Create Warp arrays
    polygon_list_wp = wp.array(all_vertices, dtype=wp.float32, device="cuda")
    polygon_indices_wp = wp.array(polygon_indices, dtype=wp.int32, device="cuda")
    results_wp = wp.zeros(num_rays, dtype=wp.float32, device="cuda")
    indices_wp = wp.zeros(num_rays, dtype=wp.int32, device="cuda")

    # Launch kernel
    wp.launch(
        faux_ray_kernel,
        dim=num_rays,
        inputs=[
            polygon_list_wp,
            polygon_indices_wp,
            len(polygons),
            float(origin[0]),
            float(origin[1]),
            float(angle_start),
            float(angle_inc),
            num_rays,
            float(max_range),
            float(resolution),
            results_wp,
            indices_wp,
        ],
    )

    return results_wp.numpy(), indices_wp.numpy()
