"""
This module provides functions for performing various geometric and visibility checks using CUDA for parallel computation.

Functions:
    contains(poly, points):
        Checks if a set of points are inside a given polygon using the winding number algorithm.

    visibility(data, start, ends):
        Computes visibility from a start point to multiple end points on a grid using Bresenham's line algorithm.

    visibility_from_region(data, starts, ends):
        Computes visibility from multiple start points to multiple end points on a grid using Bresenham's line algorithm.

    visibility_from_real_region(data, origin, resolution, starts, ends):
        Computes visibility from multiple start points to multiple end points on a grid with real-world coordinates using a floating-point grid traversal algorithm.

    faux_scan(polygons, origin, angle_start, angle_inc, num_rays, max_range, resolution):
        Performs a faux laser scan from an origin point, simulating rays at specified angles and increments, and checking for intersections with polygons.
"""

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

# import pycuda.autoprimaryctx
# import pycuda.autoinit # Removed to allow flexible context management
from pycuda.compiler import SourceModule

import numpy as np
import atexit  # For ensuring context cleanup on exit

# --- CUDA Context Management ---
_polycheck_module_context = (
    None  # Stores the context active during SourceModule compilation
)
_polycheck_created_this_context = (
    False  # True if polycheck created _polycheck_module_context
)


def _establish_module_context():
    """
    Ensures a CUDA context is available for SourceModule compilation and sets
    _polycheck_module_context. Called once when the module is loaded.
    Prefers an existing current context. If none, creates one on device 0.
    """
    global _polycheck_module_context, _polycheck_created_this_context
    # This function should only effectively run once at module import.
    if _polycheck_module_context is not None:
        return

    # Try to get an already current context
    try:
        _polycheck_module_context = cuda.Context.get_current()
        # If successful, _polycheck_created_this_context remains False,
        # as this module did not create this context.
    except cuda.LogicError:  # No current context
        cuda.init()  # Ensure CUDA driver is initialized
        device = cuda.Device(0)  # Default to device 0
        _polycheck_module_context = device.make_context()  # Creates and makes current
        _polycheck_created_this_context = (
            True  # Mark that polycheck created this context
        )


_establish_module_context()  # Ensure context is ready for SourceModule compilation


def _cleanup_polycheck_context_atexit():
    """
    Registered with atexit. Cleans up the CUDA context created by polycheck
    at module load time, if one was indeed created by this module because
    no other context was active at that time.
    """
    global _polycheck_module_context, _polycheck_created_this_context
    if _polycheck_created_this_context and _polycheck_module_context:
        try:
            # A CUDA context must be popped from the current thread's context stack
            # before it can be detached.
            is_current_on_this_thread = False
            try:
                if cuda.Context.get_current() == _polycheck_module_context:
                    is_current_on_this_thread = True
            except cuda.LogicError:  # No context is current on this thread.
                pass

            if is_current_on_this_thread:
                _polycheck_module_context.pop()

            _polycheck_module_context.detach()  # Destroy the context

            _polycheck_module_context = None
            _polycheck_created_this_context = False
        except cuda.Error:
            # Suppress errors during atexit cleanup (e.g., if context was already destroyed)
            pass
        except Exception:
            # Suppress any other unexpected errors during atexit cleanup
            pass


atexit.register(_cleanup_polycheck_context_atexit)
# --- End CUDA Context Management ---

BLOCK_SIZE = 32
Y_BLOCK_SIZE = 32
MAX_BLOCKS = 32
mod = SourceModule(
    """

    #include <cmath>
    #include <cfloat>

    //
    // Based on a comment from the following link on checking for zero:
    //
    // https://forums.developer.nvidia.com/t/on-tackling-float-point-precision-issues-in-cuda/79060
    //
    __device__
    inline bool is_zero(float f){
        return f >= -FLT_EPSILON && f <= FLT_EPSILON;
    }

    __device__
    inline bool is_equal(float f1, float f2){
        return fabs(f1 - f2) < FLT_EPSILON;
    }



    __device__ int
    epsilon_round(float value) {
        const float epsilon = 2e-6;

        float rounded_value = roundf(value);
        if( fabs(value - rounded_value) < epsilon ) {
            return static_cast<int>(rounded_value);
        } else {
            return static_cast<int>(value);
        }
    }


    // side( v1, v2, point ): checks the position of a point relative to a (directional) infinite line.
    //
    // @param: v1: starting vertex of the line
    // @param: v2: ending vertex, indicating the direction the line is headed
    // @param: point: the point to check
    // @return: +ve if point is left of v1->v2, 0 if on the line, -ve otherwise
    //
    inline __device__ auto
    side(float v1x, float v1y, float v2x, float v2y, float px, float py) -> float {
        return (v2x - v1x) * (py - v1y) - (px - v1x) * (v2y - v1y);
    }

    __device__
    bool test_point(const float *polygon_data, const int num_vertices, float px, float py ) {
        auto winding_number = 0;
        for (int vertex = 0; vertex < num_vertices; vertex++) {
            auto v1x = polygon_data[vertex * 2];
            auto v1y = polygon_data[vertex * 2 + 1];
            auto v2x = polygon_data[((vertex + 1) % num_vertices) * 2];
            auto v2y = polygon_data[((vertex + 1) % num_vertices) * 2 + 1];
            if (v1y <= py) {
                if (v2y > py) {
                    if (side(v1x, v1y, v2x, v2y, px, py) > 0) {
                        winding_number += 1;
                    }
                }
            } else {
                if (v2y <= py) {
                    if (side(v1x, v1y, v2x, v2y, px, py) < 0) {
                        winding_number -= 1;
                    }
                }
            }
        }
        return winding_number != 0 ? true : false;
    }

    __global__
    void check_points(const float *polygon_data, int num_vertices,
                      const float *points_data, int num_points,
                      float *result_data) {

        auto start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (int i = start_index; i < num_points; i += stride) {
            auto px = points_data[i * 2];
            auto py = points_data[i * 2 + 1];

            result_data[i] = test_point( polygon_data, num_vertices, px, py );
        }
    }

    __device__ float
    line_real_observation( const float* data, int height, int width, float origin_x, float origin_y,
                           float resolution, float src_x, float src_y, float end_x, float end_y ) {

        // Implement floating point fast grid traversal based on the work of Amanatides and Woo
        // (http://www.cse.yorku.ca/~amana/research/grid.pdf)
        float dx = end_x - src_x;
        float dy = end_y - src_y;
        float magnitude = sqrt(dx*dx + dy*dy);
        // Handle zero magnitude case (src == end or very close)
        if (is_zero(magnitude)) {
            auto s_check_x = static_cast<int>(floorf((src_x - origin_x) / resolution));
            auto s_check_y = static_cast<int>(floorf((src_y - origin_y) / resolution));
            if (s_check_x < 0 || s_check_x >= width || s_check_y < 0 || s_check_y >= height) return 0.0f;
            return 1.0f; // Start and end are same, and inside grid: visible by default
        }
        dx /= (magnitude);
        dy /= (magnitude);

        auto rx = ( src_x - origin_x ) / resolution;
        auto sx = static_cast<int>(floorf(rx));
        auto ry = ( src_y - origin_y ) / resolution;
        auto sy = static_cast<int>(floorf(ry));

        // return 0 if the point is outside the grid.
        if( sx < 0 || sx >= width || sy < 0 || sy >= height ) {
            return 0.0f;
        }

        auto ex = static_cast<int>(floorf(( end_x - origin_x ) / resolution));
        auto ey = static_cast<int>(floorf(( end_y - origin_y ) / resolution));

        if( sx == ex && sy == ey ) {
            return 1.0f;
        }

        int step_x;
        float t_max_x;
        float t_delta_x;

        if (is_zero(dx)) {
            step_x = 0;
            t_max_x = FLT_MAX;
            t_delta_x = FLT_MAX;
        } else if( dx > 0.0f ) {
            step_x = 1;
            t_max_x = (floorf(rx) + 1.0f - rx) / dx;
            t_delta_x = 1.0f / dx;
        } else { // dx < 0.0f
            step_x = -1;
            t_max_x = (rx - floorf(rx)) / (-dx);
            t_delta_x = 1.0f / (-dx);
        }

        int step_y;
        float t_max_y;
        float t_delta_y;

        if (is_zero(dy)) {
            step_y = 0;
            t_max_y = FLT_MAX;
            t_delta_y = FLT_MAX;
        } else if( dy > 0.0f ) {
            step_y = 1;
            t_max_y = (floorf(ry) + 1.0f - ry) / dy;
            t_delta_y = 1.0f / dy;
        } else { // dy < 0.0f
            step_y = -1;
            t_max_y = (ry - floorf(ry)) / (-dy);
            t_delta_y = 1.0f / (-dy);
        }


        auto observation = 1.0f;    // assume the point is initially viewable
        while( true ) {
            if( t_max_x < t_max_y ) {
                sx += step_x;
                t_max_x += t_delta_x;
            } else {
                sy += step_y;
                t_max_y += t_delta_y;
            }

            // reached the target, don't include the target point in the observation
            if( sx == ex && sy == ey ) {
                break;
            }

            // check for grid boundaries
            if( sx < 0 || sx >= width || sy < 0 || sy >= height ) {
                observation = 0.0f;
                break;
            }

            // If we haven't reached the end of the line, apply the view probability to the current observation
            observation *= (1.0f - data[ sy * width + sx]);
            if( is_zero(observation) ) {          // early stopping condition
                observation = 0.0f; // Ensure it's exactly zero
                break;
            }
        }

        return observation;
    }


    __device__ float
    line_observation( const float* data, int height, int width, int sx, int sy, int ex, int ey ) {
        // Using Bresenham implementation based on description found at:
        //   http://members.chello.at/~easyfilter/Bresenham.pdf
        auto dx = abs(sx-ex);
        auto step_x = sx < ex ? 1 : -1;
        auto dy = -abs(sy-ey);
        auto step_y = sy < ey ? 1 : -1;
        auto error = dx + dy;

        if( sx == ex && sy == ey ) {
            return 1.0;
        }

        auto observation = 1.0;    // assume the point is initially viewable
        for( ;; ) {
            auto e2 = 2 * error;
            if( e2 >= dy ) {
                if( sx == ex ) {
                    break;
                }
                error += dy;
                sx += step_x;
            }
            if( e2 <= dx ) {
                if( sy == ey ) {
                    break;
                }
                error += dx;
                sy += step_y;
            }

            if( sx == ex && sy == ey ) {
                break;
            }

            // If we haven't reached the end of the line, apply the view probability to the current observation
            observation *= (1.0 - data[ sy * width + sx]);
            if( observation < FLT_EPSILON*2 ) {          // early stopping condition
                observation = 0;
                break;
            }
        }

        return observation;
    }

    __device__ float
    line_range( const float *polygon_list, const int * polygon_indices, int num_polygons, float sx, float sy, float angle, float max_range, float resolution) {

        float ex, ey;
        auto x_inc = cos(angle) * resolution;
        auto y_inc = sin(angle) * resolution;
        auto dist = 0.0;

        ex = sx;
        ey = sy;
        while( dist < max_range ) {
            ex += x_inc;
            ey += y_inc;

            for( int i = 0; i < num_polygons; i++ ) {
                if( test_point( &polygon_list[ polygon_indices[i] * 2], size_t(polygon_indices[i+1] - polygon_indices[i]), ex, ey ) ) {
                    // in the polygon -- return the current length
                    return( dist );
                }
            }

            dist += resolution;
        }

        // If we reach this point, the ray contacted nothing in flight
        return -1.0;
    }

    __global__ void
    check_visibility(const float *data, const int height, const int width, const int *start,
                          const int *ends, int num_ends, float *results ) {

        auto start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (auto i = start_index; i < num_ends; i += stride) {
            int ex, ey;
            ex = ends[i*2];
            ey = ends[i*2+1];
            results[ey*width + ex] = line_observation( data, height, width, start[0], start[1], ex, ey );
        }
    }

    __global__ void
    check_real_visibility(const float *data, const int height, const int width,
                          const float origin_x, const float origin_y, const float resolution,
                          const float *start, const float *ends, int num_ends, float *results ) {

        auto start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (auto i = start_index; i < num_ends; i += stride) {
            int ex, ey;
            ex = ends[i*2];
            ey = ends[i*2+1];
            results[ey*width + ex] = line_real_observation( data, height, width, origin_x, origin_y, resolution, start[0], start[1], ex, ey );
        }
    }


    __global__ void
    check_region_visibility(const float *data, const int height, const int width, const int *starts, int num_starts,
                          const int *ends, int num_ends, float *results ) {

        auto ends_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto ends_stride = blockDim.x * gridDim.x;
        auto starts_index = blockIdx.y * blockDim.y + threadIdx.y;
        auto starts_stride = blockDim.y * gridDim.y;

        for (auto si = starts_index; si < num_starts; si += starts_stride) {
            auto sx = starts[si*2];
            auto sy = starts[si*2+1];

            for (auto ei = ends_index; ei < num_ends; ei += ends_stride) {
                auto ex = ends[ei*2];
                auto ey = ends[ei*2+1];
                results[si * num_ends + ei] = line_observation(data, height, width, sx, sy, ex, ey);
            }
        }
    }


    __global__ void
    check_real_region_visibility(const float *data, const int height, const int width,
                                 const float origin_x, const float origin_y, const float resolution,
                                 const float *starts, int num_starts, const float *ends, int num_ends, float *results ) {

        auto ends_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto ends_stride = blockDim.x * gridDim.x;
        auto starts_index = blockIdx.y * blockDim.y + threadIdx.y;
        auto starts_stride = blockDim.y * gridDim.y;

        for (auto si = starts_index; si < num_starts; si += starts_stride) {
            auto sx = starts[si*2];
            auto sy = starts[si*2+1];

            for (auto ei = ends_index; ei < num_ends; ei += ends_stride) {
                auto ex = ends[ei*2];
                auto ey = ends[ei*2+1];
                results[si * num_ends + ei] = line_real_observation(data, height, width, origin_x, origin_y,
                                                                    resolution, sx, sy, ex, ey);
            }
        }
    }

    __global__ void
    faux_ray(const float *polygon_list, const int* polygon_indices, int num_polygons, float start_x, float start_y,
             const float angle_start, const float angle_increment, const int num_rays, const float max_range, const float resolution,
             float *results ) {

        auto start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (auto i = start_index; i < num_rays; i += stride) {
            auto angle = angle_start + i * angle_increment;
            results[i] = line_range(polygon_list, polygon_indices, num_polygons, start_x, start_y, angle, max_range, resolution );
        }
    }
"""
)


def contains(poly, points):
    num_vertices = len(poly)
    num_points = len(points)

    poly = poly.astype(np.float32)
    points = points.astype(np.float32)

    polygon_size = poly.nbytes
    points_size = points.nbytes
    results_size = points_size // 2

    # allocate device memory for the polygon and the points to test, and copy the data over
    poly_gpu = cuda.mem_alloc(polygon_size)
    cuda.memcpy_htod(poly_gpu, poly)
    points_gpu = cuda.mem_alloc(points_size)
    cuda.memcpy_htod(points_gpu, points)
    results_gpu = cuda.mem_alloc(results_size)

    func = mod.get_function("check_points")
    # block_size = 256
    # num_blocks = max(128, int((num_points + block_size - 1) / block_size))
    block_size = 32
    num_blocks = 32
    func(
        poly_gpu,
        np.int32(num_vertices),
        points_gpu,
        np.int32(num_points),
        results_gpu,
        block=(block_size, num_blocks, 1),
    )

    # copy the results back
    results = np.zeros(
        [
            num_points,
        ],
        dtype=np.float32,
    )
    cuda.memcpy_dtoh(results, results_gpu)

    return results


def initialize_cuda_context(device_id=0):
    """
    Creates and activates a new CUDA context on the specified device.

    This function provides a way for users to explicitly create a new,
    regular (non-primary) CUDA context. The newly created context will
    become the current context on this thread. This is useful if the user
    wants to manage contexts explicitly or ensure operations run on a
    specific device within a fresh context.

    This method avoids the issues associated with `pycuda.autoinit`'s
    management of primary contexts, thus being less "disturbing" to
    an environment where contexts might already be managed.

    Args:
        device_id (int): The ID of the CUDA device on which to create the context.

    Returns:
        pycuda.driver.Context: The newly created and activated CUDA context.

    Note:
        The polycheck module's internal CUDA kernels (compiled into `mod`)
        are associated with the CUDA context that was active when the
        `polycheck.polycheck` module was first imported. If this function
        is called *after* the module has been loaded and it establishes a
        *different* context, ensure that subsequent calls to polycheck
        functions are compatible with this context change. PyCUDA operations
        (memory allocation, kernel launches) occur in the currently active context.
    """
    cuda.init()  # Ensure CUDA driver is initialized
    device = cuda.Device(device_id)
    context = device.make_context()  # Creates and makes the context current
    return context


def visibility(data, start, ends):
    data = data.astype(np.float32)
    height, width = data.shape
    data_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(data_gpu, data)

    start = start.astype(np.int32)
    start_gpu = cuda.mem_alloc(start.nbytes)
    cuda.memcpy_htod(start_gpu, start)

    ends = ends.astype(np.int32)
    ends_gpu = cuda.mem_alloc(ends.nbytes)
    cuda.memcpy_htod(ends_gpu, ends)
    num_ends = len(ends)

    results_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memset_d8(results_gpu, 0, data.nbytes)

    func = mod.get_function("check_visibility")
    # block_size = 256
    # num_blocks = max(128, int((num_points + block_size - 1) / block_size))
    block_size = BLOCK_SIZE
    num_blocks = MAX_BLOCKS
    func(
        data_gpu,
        np.int32(height),
        np.int32(width),
        start_gpu,
        ends_gpu,
        np.int32(num_ends),
        results_gpu,
        block=(block_size, num_blocks, 1),
    )

    # copy the results back
    results = np.zeros_like(data, dtype=np.float32)
    cuda.memcpy_dtoh(results, results_gpu)

    return results


def visibility_from_region(data, starts, ends):
    data = data.astype(np.float32)
    height, width = data.shape
    data_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(data_gpu, data)

    starts = np.array(starts, dtype=np.int32)
    num_starts = len(starts)
    starts_gpu = cuda.mem_alloc(starts.nbytes)
    cuda.memcpy_htod(starts_gpu, starts)

    ends = np.array(ends, dtype=np.int32)
    ends_gpu = cuda.mem_alloc(ends.nbytes)
    cuda.memcpy_htod(ends_gpu, ends)
    num_ends = len(ends)

    results_size = num_starts * num_ends * np.float32().nbytes
    results_gpu = cuda.mem_alloc(results_size)

    # block_size = 256
    # num_blocks = max(128, int((num_points + block_size - 1) / block_size))
    block = (BLOCK_SIZE, MAX_BLOCKS, 1)
    grid = (
        max(1, min(MAX_BLOCKS, int((num_ends + BLOCK_SIZE - 1) / BLOCK_SIZE))),
        max(1, min(MAX_BLOCKS, int((num_starts + Y_BLOCK_SIZE - 1) / Y_BLOCK_SIZE))),
    )

    func = mod.get_function("check_region_visibility")
    func(
        data_gpu,
        np.int32(height),
        np.int32(width),
        starts_gpu,
        np.int32(num_starts),
        ends_gpu,
        np.int32(num_ends),
        results_gpu,
        block=block,
        grid=grid,
    )

    # copy the results back
    results = np.zeros((num_starts * num_ends), dtype=np.float32)
    cuda.memcpy_dtoh(results, results_gpu)

    return results


def visibility_from_real_region(data, origin, resolution, starts, ends):
    data = data.astype(np.float32)
    height, width = data.shape
    data_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(data_gpu, data)

    starts = np.array(starts, dtype=np.float32)
    num_starts = len(starts)
    starts_gpu = cuda.mem_alloc(starts.nbytes)
    cuda.memcpy_htod(starts_gpu, starts)

    ends = np.array(ends, dtype=np.float32)
    ends_gpu = cuda.mem_alloc(ends.nbytes)
    cuda.memcpy_htod(ends_gpu, ends)
    num_ends = len(ends)

    results_size = num_starts * num_ends * np.float32().nbytes
    results_gpu = cuda.mem_alloc(results_size)

    # block_size = 256
    # num_blocks = max(128, int((num_points + block_size - 1) / block_size))
    block = (BLOCK_SIZE, MAX_BLOCKS, 1)
    grid = (
        max(1, min(MAX_BLOCKS, int((num_ends + BLOCK_SIZE - 1) / BLOCK_SIZE))),
        max(1, min(MAX_BLOCKS, int((num_starts + Y_BLOCK_SIZE - 1) / Y_BLOCK_SIZE))),
    )

    func = mod.get_function("check_real_region_visibility")
    func(
        data_gpu,
        np.int32(height),
        np.int32(width),
        np.float32(origin[0]),
        np.float32(origin[1]),
        np.float32(resolution),
        starts_gpu,
        np.int32(num_starts),
        ends_gpu,
        np.int32(num_ends),
        results_gpu,
        block=block,
        grid=grid,
    )

    # copy the results back
    results = np.zeros((num_starts * num_ends), dtype=np.float32)
    cuda.memcpy_dtoh(results, results_gpu)

    return results


def faux_scan(
    polygons, origin, angle_start, angle_inc, num_rays, max_range, resolution
):
    polygons = polygons.astype(np.float32)
    poly_data_size = polygons.nbytes

    if not len(polygons):
        return np.ones((num_rays,), dtype=np.float32) * -1.0

    index = 0
    polygon_indices = [index]
    for poly in polygons:
        index += len(poly)
        polygon_indices.append(index)
    polygon_indices = np.array(polygon_indices, dtype=np.int32)

    scan_size = num_rays * np.zeros((1,), dtype=np.float32).nbytes

    poly_gpu = cuda.mem_alloc(poly_data_size)
    cuda.memcpy_htod(poly_gpu, polygons)

    poly_index_gpu = cuda.mem_alloc(polygon_indices.nbytes)
    cuda.memcpy_htod(poly_index_gpu, polygon_indices)

    results_gpu = cuda.mem_alloc(scan_size)
    cuda.memset_d32(results_gpu, 0, num_rays)

    block = (BLOCK_SIZE, MAX_BLOCKS, 1)

    func = mod.get_function("faux_ray")
    func(
        poly_gpu,
        poly_index_gpu,
        np.int32(len(polygons)),
        np.float32(origin[0]),
        np.float32(origin[1]),
        np.float32(angle_start),
        np.float32(angle_inc),
        np.int32(num_rays),
        np.float32(max_range),
        np.float32(resolution),
        results_gpu,
        block=block,
    )

    # copy the results back
    results = np.zeros(num_rays, dtype=np.float32)
    cuda.memcpy_dtoh(results, results_gpu)

    return results
