"""
This module provides functions for performing various geometric and visibility checks using CUDA for parallel computation.

Functions:
    contains(poly, points):
        Checks if a set of points are inside a given polygon using the winding number algorithm.

    visibility(data, start, ends, max_range=None):
        Computes visibility from a start point to multiple end points on a grid using Bresenham's line algorithm.

    visibility_from_region(data, starts, ends, max_range=None):
        Computes visibility from multiple start points to multiple end points on a grid using Bresenham's line algorithm.

    visibility_from_real_region(data, origin, resolution, starts, ends, max_range=None):
        Computes visibility from multiple start points to multiple end points on a grid with real-world coordinates using a floating-point grid traversal algorithm.

    sensor_visibility_from_region(data, sensors):
        Computes per-sensor clear probabilities over all grid cells using Bresenham rays and sensor range/FOV constraints.

    sensor_visibility_from_real_region(data, origin, resolution, sensors):
        Computes per-sensor clear probabilities over all grid cells using floating-point grid traversal and sensor range/FOV constraints.

    faux_scan(polygons, origin, angle_start, angle_inc, num_rays, max_range, resolution):
        Performs a faux laser scan from an origin point, simulating rays at specified angles and increments, and checking for intersections with polygons.

    initialize_cuda_context(device_id=0):
        Creates and activates a new CUDA context on the specified device.
"""

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

import pycuda.autoprimaryctx

# import pycuda.autoinit  # Removed to allow flexible context management
from pycuda.compiler import SourceModule

import numpy as np
import atexit  # For ensuring context cleanup on exit

# # --- CUDA Context Management ---
# _polycheck_module_context = (
#     None  # Stores the context active during SourceModule compilation
# )
# _polycheck_created_this_context = (
#     False  # True if polycheck created _polycheck_module_context
# )


# def _establish_module_context():
#     """
#     Ensures a CUDA context is available for SourceModule compilation and sets
#     _polycheck_module_context. Called once when the module is loaded.
#     Prefers an existing current context. If none, creates one on device 0.
#     """
#     global _polycheck_module_context, _polycheck_created_this_context
#     # This function should only effectively run once at module import.
#     if _polycheck_module_context is not None:
#         return

#     # Try to get an already current context
#     try:
#         _polycheck_module_context = cuda.Context.get_current()
#         # If successful, _polycheck_created_this_context remains False,
#         # as this module did not create this context.
#     except cuda.LogicError:  # No current context
#         cuda.init()  # Ensure CUDA driver is initialized
#         device = cuda.Device(0)  # Default to device 0
#         _polycheck_module_context = device.make_context()  # Creates and makes current
#         _polycheck_created_this_context = (
#             True  # Mark that polycheck created this context
#         )


# _establish_module_context()  # Ensure context is ready for SourceModule compilation


# def _cleanup_polycheck_context_atexit():
#     """
#     Registered with atexit. Cleans up the CUDA context created by polycheck
#     at module load time, if one was indeed created by this module because
#     no other context was active at that time.
#     """
#     global _polycheck_module_context, _polycheck_created_this_context
#     if _polycheck_created_this_context and _polycheck_module_context:
#         try:
#             # A CUDA context must be popped from the current thread's context stack
#             # before it can be detached.
#             is_current_on_this_thread = False
#             try:
#                 if cuda.Context.get_current() == _polycheck_module_context:
#                     is_current_on_this_thread = True
#             except cuda.LogicError:  # No context is current on this thread.
#                 pass

#             if is_current_on_this_thread:
#                 _polycheck_module_context.pop()

#             _polycheck_module_context.detach()  # Destroy the context

#             _polycheck_module_context = None
#             _polycheck_created_this_context = False
#         except cuda.Error:
#             # Suppress errors during atexit cleanup (e.g., if context was already destroyed)
#             pass
#         except Exception:
#             # Suppress any other unexpected errors during atexit cleanup
#             pass


# atexit.register(_cleanup_polycheck_context_atexit)
# # --- End CUDA Context Management ---

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

    constexpr float PI_F = 3.14159265358979323846f;
    constexpr float TWO_PI_F = 6.28318530717958647692f;

    __device__
    inline float clamp01(float value){
        return fminf(1.0f, fmaxf(0.0f, value));
    }

    __device__
    inline float wrap_to_pi(float angle){
        while (angle > PI_F) {
            angle -= TWO_PI_F;
        }
        while (angle < -PI_F) {
            angle += TWO_PI_F;
        }
        return angle;
    }

    __device__
    inline bool in_sensor_fov(float sx, float sy, float tx, float ty, float range, float direction, float fov){
        auto dx = tx - sx;
        auto dy = ty - sy;
        auto distance = sqrtf(dx * dx + dy * dy);

        if (range > 0.0f && distance > range) {
            return false;
        }

        if (fov > 0.0f && fov < (TWO_PI_F - 1e-6f)) {
            if (is_zero(dx) && is_zero(dy)) {
                return true;
            }
            auto bearing = atan2f(dy, dx);
            auto angle_delta = fabsf(wrap_to_pi(bearing - direction));
            if (angle_delta > (0.5f * fov)) {
                return false;
            }
        }

        return true;
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
                           float resolution, float src_x, float src_y, float end_x, float end_y, float max_range ) {

        // Implement floating point fast grid traversal based on the work of Amanatides and Woo
        // (http://www.cse.yorku.ca/~amana/research/grid.pdf)
        float dx = end_x - src_x;
        float dy = end_y - src_y;
        float magnitude = sqrt(dx*dx + dy*dy);

        // If the target is beyond max_range, it's not visible.
        if (max_range > 0.0f && magnitude > max_range) {
            return 0.0f;
        }

        // Handle zero magnitude case (src == end or very close)
        if (is_zero(magnitude)) {
            auto s_check_x = static_cast<int>(floorf((src_x - origin_x) / resolution));
            auto s_check_y = static_cast<int>(floorf((src_y - origin_y) / resolution));
            if (s_check_x < 0 || s_check_x >= width || s_check_y < 0 || s_check_y >= height) {
                return 0.0f;
            }
            return 1.0f;
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

        // Calculate the distance to the center of the target cell for a robust termination check
        // float end_center_x = (ex + 0.5f) * resolution + origin_x;
        // float end_center_y = (ey + 0.5f) * resolution + origin_y;
        // float dist_to_end_center = sqrtf(powf(end_center_x - src_x, 2) + powf(end_center_y - src_y, 2));

        int step_x;
        float t_max_x;
        float t_delta_x;

        if (is_zero(dx)) {
            step_x = 0;
            t_max_x = FLT_MAX;
            t_delta_x = FLT_MAX;
        } else if( dx > 0.0f ) {
            step_x = 1;
            t_max_x = (floorf(rx) + 1.0f - rx) * resolution / dx;
            t_delta_x = resolution / dx;
        } else { // dx < 0.0f
            step_x = -1;
            t_max_x = (rx - floorf(rx)) * resolution / (-dx);
            t_delta_x = resolution / (-dx);
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
            t_max_y = (floorf(ry) + 1.0f - ry) * resolution / dy;
            t_delta_y = resolution / dy;
        } else { // dy < 0.0f
            step_y = -1;
            t_max_y = (ry - floorf(ry)) * resolution / (-dy);
            t_delta_y = resolution / (-dy);
        }

        auto observation = 1.0f;    // assume the point is initially viewable

        while( true ) {

            // Robust termination check: stop if we've passed the length of the line
            if (fminf(t_max_x, t_max_y) > magnitude ) { // dist_to_end_center) {
                break;
            }

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
    line_observation( const float* data, int height, int width, int sx, int sy, int ex, int ey, int max_range = 0 ) {
        // Using Bresenham implementation based on description found at:
        //   http://members.chello.at/~easyfilter/Bresenham.pdf
        auto dx = abs(sx-ex);
        auto step_x = sx < ex ? 1 : -1;
        auto dy = -abs(sy-ey);
        auto step_y = sy < ey ? 1 : -1;
        auto error = dx + dy;
        auto steps = 0;

        if( sx == ex && sy == ey ) {
            return 1.0;
        }

        auto observation = 1.0;    // assume the point is initially viewable
        for( ;; ) {
            // Exclude endpoint: stop as soon as we are on it.
            if( sx == ex && sy == ey ) {
                break;
            }

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

            steps += 1;
            if( max_range > 0 && steps > max_range ) {
                observation = 0;
                break; // stop if we exceed the max_range
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
    line_observation_sum(const float* data, int height, int width, int sx, int sy, int ex, int ey) {
        if (sx < 0 || sx >= width || sy < 0 || sy >= height) {
            return 0.0f;
        }
        if (ex < 0 || ex >= width || ey < 0 || ey >= height) {
            return 0.0f;
        }
        if (sx == ex && sy == ey) {
            return 1.0f;
        }

        auto dx = abs(sx - ex);
        auto step_x = sx < ex ? 1 : -1;
        auto dy = -abs(sy - ey);
        auto step_y = sy < ey ? 1 : -1;
        auto error = dx + dy;

        auto blocked_sum = 0.0f;
        for (;;) {
            if (sx == ex && sy == ey) {
                break;
            }

            auto e2 = 2 * error;
            if (e2 >= dy) {
                if (sx == ex) {
                    break;
                }
                error += dy;
                sx += step_x;
            }
            if (e2 <= dx) {
                if (sy == ey) {
                    break;
                }
                error += dx;
                sy += step_y;
            }

            // Exclude destination from blocking sum.
            if (sx == ex && sy == ey) {
                break;
            }

            if (sx < 0 || sx >= width || sy < 0 || sy >= height) {
                return 0.0f;
            }

            blocked_sum += data[sy * width + sx];
            if (blocked_sum >= 1.0f) {
                return 0.0f;
            }
        }

        return clamp01(1.0f - blocked_sum);
    }

    __device__ float
    line_real_observation_sum(const float* data, int height, int width, float origin_x, float origin_y,
                        float resolution, float src_x, float src_y, float end_x, float end_y, float max_range) {
        auto dx = end_x - src_x;
        auto dy = end_y - src_y;
        auto magnitude = sqrtf(dx * dx + dy * dy);

        if (max_range > 0.0f && magnitude > max_range) {
            return 0.0f;
        }

        if (is_zero(magnitude)) {
            auto s_check_x = static_cast<int>(floorf((src_x - origin_x) / resolution));
            auto s_check_y = static_cast<int>(floorf((src_y - origin_y) / resolution));
            if (s_check_x < 0 || s_check_x >= width || s_check_y < 0 || s_check_y >= height) {
                return 0.0f;
            }
            return 1.0f;
        }

        dx /= magnitude;
        dy /= magnitude;

        auto rx = (src_x - origin_x) / resolution;
        auto sx = static_cast<int>(floorf(rx));
        auto ry = (src_y - origin_y) / resolution;
        auto sy = static_cast<int>(floorf(ry));

        if (sx < 0 || sx >= width || sy < 0 || sy >= height) {
            return 0.0f;
        }

        auto ex = static_cast<int>(floorf((end_x - origin_x) / resolution));
        auto ey = static_cast<int>(floorf((end_y - origin_y) / resolution));

        if (ex < 0 || ex >= width || ey < 0 || ey >= height) {
            return 0.0f;
        }

        if (sx == ex && sy == ey) {
            return 1.0f;
        }

        int step_x;
        float t_max_x;
        float t_delta_x;

        if (is_zero(dx)) {
            step_x = 0;
            t_max_x = FLT_MAX;
            t_delta_x = FLT_MAX;
        } else if (dx > 0.0f) {
            step_x = 1;
            t_max_x = (floorf(rx) + 1.0f - rx) * resolution / dx;
            t_delta_x = resolution / dx;
        } else {
            step_x = -1;
            t_max_x = (rx - floorf(rx)) * resolution / (-dx);
            t_delta_x = resolution / (-dx);
        }

        int step_y;
        float t_max_y;
        float t_delta_y;

        if (is_zero(dy)) {
            step_y = 0;
            t_max_y = FLT_MAX;
            t_delta_y = FLT_MAX;
        } else if (dy > 0.0f) {
            step_y = 1;
            t_max_y = (floorf(ry) + 1.0f - ry) * resolution / dy;
            t_delta_y = resolution / dy;
        } else {
            step_y = -1;
            t_max_y = (ry - floorf(ry)) * resolution / (-dy);
            t_delta_y = resolution / (-dy);
        }

        auto blocked_sum = 0.0f;

        while (true) {
            if (fminf(t_max_x, t_max_y) > magnitude) {
                break;
            }

            if (t_max_x < t_max_y) {
                sx += step_x;
                t_max_x += t_delta_x;
            } else {
                sy += step_y;
                t_max_y += t_delta_y;
            }

            // Exclude destination from blocking sum.
            if (sx == ex && sy == ey) {
                break;
            }

            if (sx < 0 || sx >= width || sy < 0 || sy >= height) {
                return 0.0f;
            }

            blocked_sum += data[sy * width + sx];
            if (blocked_sum >= 1.0f) {
                return 0.0f;
            }
        }

        return clamp01(1.0f - blocked_sum);
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
                          const int *ends, const int num_ends, const int max_range, float *results ) {

        auto start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (auto i = start_index; i < num_ends; i += stride) {
            int ex, ey;
            ex = ends[i*2];
            ey = ends[i*2+1];
            results[ey*width + ex] = line_observation( data, height, width, start[0], start[1], ex, ey, max_range );
        }
    }

    __global__ void
    check_real_visibility(const float *data, const int height, const int width,
                          const float origin_x, const float origin_y, const float resolution,
                          const float *start, const float *ends, const int num_ends, const float max_range, float *results ) {

        auto start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (auto i = start_index; i < num_ends; i += stride) {
            int ex, ey;
            ex = ends[i*2];
            ey = ends[i*2+1];
            results[ey*width + ex] = line_real_observation( data, height, width, origin_x, origin_y, resolution, start[0], start[1], ex, ey, max_range);
        }
    }


    __global__ void
    check_region_visibility(const float *data, const int height, const int width, const int *starts, const int num_starts,
                          const int *ends, const int num_ends, const int max_range, float *results ) {

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
                results[si * num_ends + ei] = line_observation(data, height, width, sx, sy, ex, ey, max_range);
            }
        }
    }


    __global__ void
    check_real_region_visibility(const float *data, const int height, const int width,
                                 const float origin_x, const float origin_y, const float resolution,
                                 const float *starts, const int num_starts, const float *ends,
                                 const int num_ends, const float max_range, float *results
                                 ) {

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
                                                                    resolution, sx, sy, ex, ey, max_range);
            }
        }
    }

    __global__ void
    check_sensor_region_visibility(const float *data, const int height, const int width,
                                   const float *sensors, const int num_sensors, float *results) {

        auto cell_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto cell_stride = blockDim.x * gridDim.x;
        auto sensor_index = blockIdx.y * blockDim.y + threadIdx.y;
        auto sensor_stride = blockDim.y * gridDim.y;
        auto num_cells = height * width;

        for (auto si = sensor_index; si < num_sensors; si += sensor_stride) {
            auto sensor_x = sensors[si * 5];
            auto sensor_y = sensors[si * 5 + 1];
            auto sensor_range = sensors[si * 5 + 2];
            auto sensor_direction = sensors[si * 5 + 3];
            auto sensor_fov = sensors[si * 5 + 4];

            auto sx = epsilon_round(sensor_x);
            auto sy = epsilon_round(sensor_y);

            for (auto ei = cell_index; ei < num_cells; ei += cell_stride) {
                auto ex = ei % width;
                auto ey = ei / width;

                if (sensor_range <= 0.0f) {
                    results[si * num_cells + ei] = (ex == sx && ey == sy) ? 1.0f : 0.0f;
                    continue;
                }

                if (!in_sensor_fov(static_cast<float>(sx), static_cast<float>(sy),
                                   static_cast<float>(ex), static_cast<float>(ey),
                                   sensor_range, sensor_direction, sensor_fov)) {
                    results[si * num_cells + ei] = 0.0f;
                    continue;
                }

                results[si * num_cells + ei] = line_observation_sum(data, height, width, sx, sy, ex, ey);
            }
        }
    }

    __global__ void
    check_sensor_real_region_visibility(const float *data, const int height, const int width,
                                        const float origin_x, const float origin_y, const float resolution,
                                        const float *sensors, const int num_sensors, float *results) {

        auto cell_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto cell_stride = blockDim.x * gridDim.x;
        auto sensor_index = blockIdx.y * blockDim.y + threadIdx.y;
        auto sensor_stride = blockDim.y * gridDim.y;
        auto num_cells = height * width;

        for (auto si = sensor_index; si < num_sensors; si += sensor_stride) {
            auto sensor_x = sensors[si * 5];
            auto sensor_y = sensors[si * 5 + 1];
            auto sensor_range = sensors[si * 5 + 2];
            auto sensor_direction = sensors[si * 5 + 3];
            auto sensor_fov = sensors[si * 5 + 4];
            auto sensor_cell_x = static_cast<int>(floorf((sensor_x - origin_x) / resolution));
            auto sensor_cell_y = static_cast<int>(floorf((sensor_y - origin_y) / resolution));

            for (auto ei = cell_index; ei < num_cells; ei += cell_stride) {
                auto ex = ei % width;
                auto ey = ei / width;
                auto target_x = origin_x + (static_cast<float>(ex) + 0.5f) * resolution;
                auto target_y = origin_y + (static_cast<float>(ey) + 0.5f) * resolution;

                if (sensor_range <= 0.0f) {
                    results[si * num_cells + ei] = (ex == sensor_cell_x && ey == sensor_cell_y) ? 1.0f : 0.0f;
                    continue;
                }

                if (!in_sensor_fov(sensor_x, sensor_y, target_x, target_y,
                                   sensor_range, sensor_direction, sensor_fov)) {
                    results[si * num_cells + ei] = 0.0f;
                    continue;
                }

                results[si * num_cells + ei] = line_real_observation_sum(
                    data, height, width, origin_x, origin_y, resolution,
                    sensor_x, sensor_y, target_x, target_y, sensor_range
                );
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


def visibility(data, start, ends, max_range=None):
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

    if max_range is None:
        max_range = np.int32(0)
    else:
        max_range = np.int32(max_range)

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
        max_range,
        results_gpu,
        block=(block_size, num_blocks, 1),
    )

    # copy the results back
    results = np.zeros_like(data, dtype=np.float32)
    cuda.memcpy_dtoh(results, results_gpu)

    return results


def visibility_from_region(data, starts, ends, max_range=None):
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

    if max_range is None:
        max_range = np.int32(0)
    else:
        max_range = np.int32(max_range)

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
        max_range,
        results_gpu,
        block=block,
        grid=grid,
    )

    # copy the results back
    results = np.zeros((num_starts * num_ends), dtype=np.float32)
    cuda.memcpy_dtoh(results, results_gpu)

    return results


def visibility_from_real_region(data, origin, resolution, starts, ends, max_range=None):
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

    if max_range is None:
        max_range = np.float32(0.0)
    else:
        max_range = np.float32(max_range)

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
        max_range,
        results_gpu,
        block=block,
        grid=grid,
    )

    # copy the results back
    results = np.zeros((num_starts * num_ends), dtype=np.float32)
    cuda.memcpy_dtoh(results, results_gpu)

    return results


def _validate_sensors(sensors):
    sensors = np.asarray(sensors, dtype=np.float32)
    if sensors.ndim != 2 or sensors.shape[1] != 5:
        raise ValueError(
            "sensors must have shape (num_sensors, 5) with columns "
            "[x, y, range, direction, fov]"
        )
    return sensors


def _wrap_to_pi_numpy(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _epsilon_round_numpy(value):
    rounded = np.rint(value)
    if np.abs(value - rounded) < 2e-6:
        return int(rounded)
    return int(value)


def _sensor_coverage_mask_grid(height, width, sensors):
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    mask = np.zeros((len(sensors), height, width), dtype=bool)

    for i, sensor in enumerate(sensors):
        sx, sy, sensor_range, direction, fov = sensor
        sx_i = _epsilon_round_numpy(float(sx))
        sy_i = _epsilon_round_numpy(float(sy))

        if sensor_range <= 0.0:
            if 0 <= sx_i < width and 0 <= sy_i < height:
                mask[i, sy_i, sx_i] = True
            continue

        dx = xx - float(sx_i)
        dy = yy - float(sy_i)
        distance = np.sqrt(dx * dx + dy * dy)
        in_range = distance <= float(sensor_range) + 1e-6

        if 0.0 < float(fov) < (2.0 * np.pi - 1e-6):
            bearing = np.arctan2(dy, dx)
            angle_delta = np.abs(_wrap_to_pi_numpy(bearing - float(direction)))
            in_fov = (distance <= 1e-8) | (angle_delta <= (0.5 * float(fov) + 1e-6))
        else:
            in_fov = np.ones((height, width), dtype=bool)

        mask[i] = in_range & in_fov

    return mask


def _sensor_coverage_mask_real(height, width, origin, resolution, sensors):
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    tx = float(origin[0]) + (xx + 0.5) * float(resolution)
    ty = float(origin[1]) + (yy + 0.5) * float(resolution)

    mask = np.zeros((len(sensors), height, width), dtype=bool)
    for i, sensor in enumerate(sensors):
        sx, sy, sensor_range, direction, fov = sensor

        if sensor_range <= 0.0:
            sensor_cell_x = int(
                np.floor((float(sx) - float(origin[0])) / float(resolution))
            )
            sensor_cell_y = int(
                np.floor((float(sy) - float(origin[1])) / float(resolution))
            )
            if 0 <= sensor_cell_x < width and 0 <= sensor_cell_y < height:
                mask[i, sensor_cell_y, sensor_cell_x] = True
            continue

        dx = tx - float(sx)
        dy = ty - float(sy)
        distance = np.sqrt(dx * dx + dy * dy)
        in_range = distance <= float(sensor_range) + 1e-6

        if 0.0 < float(fov) < (2.0 * np.pi - 1e-6):
            bearing = np.arctan2(dy, dx)
            angle_delta = np.abs(_wrap_to_pi_numpy(bearing - float(direction)))
            in_fov = (distance <= 1e-8) | (angle_delta <= (0.5 * float(fov) + 1e-6))
        else:
            in_fov = np.ones((height, width), dtype=bool)

        mask[i] = in_range & in_fov

    return mask


def _combine_sensor_observations(per_sensor_observation, coverage_mask, combine):
    if combine == "union":
        # Out-of-coverage sensors contribute 0 observation probability.
        effective = np.where(coverage_mask, per_sensor_observation, 0.0)
        combined = 1.0 - np.prod(1.0 - effective, axis=0)
    elif combine == "product":
        # Out-of-coverage sensors are neutral factors for multiplicative fusion.
        effective = np.where(coverage_mask, per_sensor_observation, 1.0)
        combined = np.prod(effective, axis=0)
        combined = np.where(np.any(coverage_mask, axis=0), combined, 0.0)
    else:
        raise ValueError("combine must be 'union' or 'product'")

    return np.clip(combined, 0.0, 1.0).astype(np.float32)


def sensor_visibility_from_region(data, sensors, combine="union"):
    """
    Compute per-sensor clear probabilities over every grid cell using Bresenham rays.

    Args:
        data (ndarray): Occupancy grid of shape (H, W), where each cell stores occupancy
            probability in [0, 1].
        sensors (array-like): Shape (M, 5) with columns [x, y, range, direction, fov].
            Positions are interpreted in grid-cell coordinates and rounded to the nearest
            integer cell for Bresenham tracing. Angles are in radians, and fov is the full
            opening angle centered on direction.

    Returns:
        tuple[ndarray, ndarray]:
            - per_sensor_observation: shape (M, H, W)
            - combined_observation: shape (H, W)
              * `combine='union'`: 1 - prod_m(1 - p_m)
              * `combine='product'`: prod_m p_m across covering sensors only
    """
    data = np.asarray(data, dtype=np.float32)
    data = np.clip(data, 0.0, 1.0)
    height, width = data.shape

    sensors = _validate_sensors(sensors)
    num_sensors = len(sensors)

    if num_sensors == 0:
        return (
            np.zeros((0, height, width), dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
        )

    data_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(data_gpu, data)

    sensors_gpu = cuda.mem_alloc(sensors.nbytes)
    cuda.memcpy_htod(sensors_gpu, sensors)

    num_cells = height * width
    results_size = num_sensors * num_cells * np.float32().nbytes
    results_gpu = cuda.mem_alloc(results_size)

    block = (BLOCK_SIZE, Y_BLOCK_SIZE, 1)
    grid = (
        max(1, min(MAX_BLOCKS, int((num_cells + BLOCK_SIZE - 1) / BLOCK_SIZE))),
        max(
            1,
            min(MAX_BLOCKS, int((num_sensors + Y_BLOCK_SIZE - 1) / Y_BLOCK_SIZE)),
        ),
    )

    func = mod.get_function("check_sensor_region_visibility")
    func(
        data_gpu,
        np.int32(height),
        np.int32(width),
        sensors_gpu,
        np.int32(num_sensors),
        results_gpu,
        block=block,
        grid=grid,
    )

    per_sensor_observation = np.zeros((num_sensors * num_cells), dtype=np.float32)
    cuda.memcpy_dtoh(per_sensor_observation, results_gpu)
    per_sensor_observation = per_sensor_observation.reshape(
        (num_sensors, height, width)
    )
    per_sensor_observation = np.clip(per_sensor_observation, 0.0, 1.0).astype(
        np.float32
    )

    coverage_mask = _sensor_coverage_mask_grid(height, width, sensors)
    combined_observation = _combine_sensor_observations(
        per_sensor_observation, coverage_mask, combine
    )

    return per_sensor_observation, combined_observation


def sensor_visibility_from_real_region(
    data, origin, resolution, sensors, combine="union"
):
    """
    Compute per-sensor clear probabilities over every grid cell using real-valued rays.

    Args:
        data (ndarray): Occupancy grid of shape (H, W), where each cell stores occupancy
            probability in [0, 1].
        origin (tuple[float, float]): World-space origin of grid cell (0, 0).
        resolution (float): Cell size in world-space units.
        sensors (array-like): Shape (M, 5) with columns [x, y, range, direction, fov] in
            world coordinates/units. Angles are in radians, and fov is the full opening
            angle centered on direction.

    Returns:
        tuple[ndarray, ndarray]:
            - per_sensor_observation: shape (M, H, W)
            - combined_observation: shape (H, W)
              * `combine='union'`: 1 - prod_m(1 - p_m)
              * `combine='product'`: prod_m p_m across covering sensors only
    """
    data = np.asarray(data, dtype=np.float32)
    data = np.clip(data, 0.0, 1.0)
    height, width = data.shape

    sensors = _validate_sensors(sensors)
    num_sensors = len(sensors)

    if num_sensors == 0:
        return (
            np.zeros((0, height, width), dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
        )

    data_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(data_gpu, data)

    sensors_gpu = cuda.mem_alloc(sensors.nbytes)
    cuda.memcpy_htod(sensors_gpu, sensors)

    num_cells = height * width
    results_size = num_sensors * num_cells * np.float32().nbytes
    results_gpu = cuda.mem_alloc(results_size)

    block = (BLOCK_SIZE, Y_BLOCK_SIZE, 1)
    grid = (
        max(1, min(MAX_BLOCKS, int((num_cells + BLOCK_SIZE - 1) / BLOCK_SIZE))),
        max(
            1,
            min(MAX_BLOCKS, int((num_sensors + Y_BLOCK_SIZE - 1) / Y_BLOCK_SIZE)),
        ),
    )

    func = mod.get_function("check_sensor_real_region_visibility")
    func(
        data_gpu,
        np.int32(height),
        np.int32(width),
        np.float32(origin[0]),
        np.float32(origin[1]),
        np.float32(resolution),
        sensors_gpu,
        np.int32(num_sensors),
        results_gpu,
        block=block,
        grid=grid,
    )

    per_sensor_observation = np.zeros((num_sensors * num_cells), dtype=np.float32)
    cuda.memcpy_dtoh(per_sensor_observation, results_gpu)
    per_sensor_observation = per_sensor_observation.reshape(
        (num_sensors, height, width)
    )
    per_sensor_observation = np.clip(per_sensor_observation, 0.0, 1.0).astype(
        np.float32
    )

    coverage_mask = _sensor_coverage_mask_real(
        height, width, origin, resolution, sensors
    )
    combined_observation = _combine_sensor_observations(
        per_sensor_observation, coverage_mask, combine
    )

    return per_sensor_observation, combined_observation


def faux_scan(
    polygons, origin, angle_start, angle_inc, num_rays, max_range, resolution
):
    if not len(polygons):
        return (
            np.ones((num_rays,), dtype=np.float32) * -1.0,
            np.ones((num_rays,), dtype=np.int32) * 0x7FFFFFFF,
        )

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
    poly_data_size = all_vertices.nbytes

    scan_size = num_rays * np.zeros((1,), dtype=np.float32).nbytes

    poly_gpu = cuda.mem_alloc(poly_data_size)
    cuda.memcpy_htod(poly_gpu, all_vertices)

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
