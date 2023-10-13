# PyCUDA Example!

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np


mod = SourceModule(
    """

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
