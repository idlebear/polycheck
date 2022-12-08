
// CUDA implementation of computational geometry problems
//
// One of many solutions to the point in poly problem -- this one an implementation based on Winding Numbers, but
// with no trig calls required.  Based on an algorithm published by Dan Sunday, 2012
//
//    https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html


#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cuda_runtime.h>

#include <cmath>
#include <vector>
#include <stdexcept>

#include "point_in_polygon.h"

namespace polycheck {

    // side( v1, v2, point ): checks the position of a point relative to a (directional) infinite line.
    //
    // @param: v1: starting vertex of the line
    // @param: v2: ending vertex, indicating the direction the line is headed
    // @param: point: the point to check
    // @return: +ve if point is left of v1->v2, 0 if on the line, -ve otherwise
    //
    inline __device__ auto
    side(double v1x, double v1y, double v2x, double v2y, double px, double py) -> double {
        return (v2x - v1x) * (py - v1y) - (px - v1x) * (v2y - v1y);
    }

    __global__
    void check_points(const double *polygon_data, const size_t num_vertices,
                      const double *points_data, const size_t num_points,
                      int *result_data) {

        auto start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (int i = start_index; i < num_points; i += stride) {
            auto px = points_data[i * 2];
            auto py = points_data[i * 2 + 1];

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
            result_data[i] = winding_number != 0 ? 1 : 0;
        }

    }

    template <class T>
    void copy_vector_to_device( const std::vector<std::vector<T>> V, void** ptr ) {
        auto rows = V.size();
        auto cols = V[0].size();

        CUDA_CALL(cudaMalloc( ptr, rows * cols * sizeof(T)));
        T* dst = (T*)*ptr;
        for( int row = 0; row < rows; row++ ) {
            CUDA_CALL(cudaMemcpy( dst + row * cols, V[row].data(), cols*sizeof(T), cudaMemcpyHostToDevice ));
        }
    }

    // contains( poly, point ): check if a point can be found within a polygon.
    std::vector<int>
    contains(const std::vector<std::vector<double>> &poly, const std::vector<std::vector<double>> &points) {

        double *polygon_data;
        double *points_data;
        int *result_data;

        auto polygon_size = poly.size() * sizeof(double) * 2;
        auto points_size = points.size() * sizeof(double) * 2;
        auto results_size = points.size() * sizeof(int);

        // allocate device memory for the polygon and the points to test
        copy_vector_to_device( poly, (void**)&polygon_data );
        copy_vector_to_device( points, (void**)&points_data);
        CUDA_CALL(cudaMalloc((void **) &result_data, results_size));
        CUDA_CALL(cudaMemset(result_data, 0, results_size));

        printf( "num points: %d\n", points.size() );
        auto block_size = BLOCK_SIZE;
        auto num_blocks = std::max(MAX_BLOCKS, int((points.size() + block_size - 1) / block_size));
        check_points<<<num_blocks, block_size>>>(polygon_data, poly.size(), points_data, points.size(), result_data);

        // wait for synchronization/completion
        CUDA_CALL(cudaDeviceSynchronize());

        // copy the results back from the device
        auto results = std::vector<int>();
        results.resize( points.size() );

        CUDA_CALL(cudaMemcpy(results.data(), result_data, results_size, cudaMemcpyDeviceToHost));

        // free everything up
        CUDA_CALL(cudaFree(polygon_data))
        CUDA_CALL(cudaFree(points_data))
        CUDA_CALL(cudaFree(result_data))

        return results;
    }

}  // polycheck namespace

