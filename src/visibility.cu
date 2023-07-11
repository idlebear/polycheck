// CUDA implementation of computational geometry problems
//
// One of many solutions to the point in poly problem -- this one an implementation based on Winding Numbers, but
// with no trig calls required.  Based on an algorithm published by Dan Sunday, 2012
//
//    https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html

#include <iostream>

#include <cuda_runtime.h>
//#include <curand_kernel.h>
//#include <memory>
#include <vector>

#include <float.h>

#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cmath>
#include <vector>
#include <stdexcept>

#include "visibility.h"


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

    __device__
    bool test_point(const double *polygon_data, const size_t num_vertices, double px, double py ) {
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
    void check_points(const double *polygon_data, const size_t num_vertices,
                      const double *points_data, const size_t num_points,
                      uint32_t *result_data) {

        auto start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (int i = start_index; i < num_points; i += stride) {
            auto px = points_data[i * 2];
            auto py = points_data[i * 2 + 1];

            result_data[i] = test_point( polygon_data, num_vertices, px, py );
        }
    }

    __device__ double
    line_observation( const double* data, int height, int width, int sx, int sy, int ex, int ey ) {
        // Using Bresenham implementation based on description found at:
        //   http://members.chello.at/~easyfilter/Bresenham.pdf
        auto dx = abs(sx-ex);
        auto step_x = sx < ex ? 1 : -1;
        auto dy = -abs(sy-ey);
        auto step_y = sy < ey ? 1 : -1;
        auto error = dx + dy;

        auto observation = 1.0;    // assume the point is initially viewable
        for( ;; ) {
            auto view_prob = (1.0 - data[ sy * width + sx]);
            auto e2 = 2 * error;
            if( e2 >= dy ) {
                if( sx == ex ) {
                    break;
                }
                error = error + dy;
                sx += step_x;
            }
            if( e2 <= dx ) {
                if( sy == ey ) {
                    break;
                }
                error += dx;
                sy += step_y;
            }

            // If we haven't reached the end of the line, apply the view probability to the current observation
            observation *= view_prob;
            if( observation < FLT_EPSILON*2 ) {          // early stopping condition
                observation = 0;
                break;
            }
        }

        return observation;
    }

    __device__ double
    line_range( const double *polygon_list, const size_t * polygon_indices, int num_polygons, double sx, double sy, double angle, double max_range, double resolution) {

        double ex, ey;
        auto x_inc = cos(angle) * resolution;
        auto y_inc = sin(angle) * resolution;
        auto dist = 0.0;

        ex = sx;
        ey = sy;
        while( dist < max_range ) {
            ex += x_inc;
            ey += y_inc;
            dist += resolution;

            for( int i = 0; i < num_polygons; i++ ) {
                if( polycheck::test_point( &polygon_list[ polygon_indices[i] * 2], size_t(polygon_indices[i+1] - polygon_indices[i]), ex, ey ) ) {
                    // in the polygon -- return the current length
                    return( dist );
                }
            }
        }

        // If we reach this point, the ray contacted nothing in flight
        return -1.0;
    }


    __global__ void
    check_visibility(const double *data, const int height, const int width, const int *start,
                          const int *ends, int num_ends, double *results ) {

        auto start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (auto i = start_index; i < num_ends; i += stride) {
            int ex, ey;
            if( ends == nullptr ) {
                ex = int(i % width);
                ey = int(i / width);
            } else {
                ex = ends[i*2];
                ey = ends[i*2+1];
            }
            results[ey*width + ex] = line_observation( data, height, width, start[0], start[1], ex, ey );
        }
    }

    __global__ void
    check_region_visibility(const double *data, const int height, const int width, const int *starts, int num_starts,
                          const int *ends, int num_ends, double *results ) {

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
    faux_ray(const double *polygon_list, const size_t* polygon_indices, int num_polygons, double start_x, double start_y,
             const double angle_start, const double angle_increment, const int num_rays, const double max_range, const double resolution,
             double *results ) {

        auto start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (auto i = start_index; i < num_rays; i += stride) {
            auto angle = angle_start + i * angle_increment;
            results[i] = line_range(polygon_list, polygon_indices, num_polygons, start_x, start_y, angle, max_range, resolution );
        }
    }

    // contains( poly, point ): check if a point can be found within a polygon.
    void contains( const double *poly_ptr, int num_vertices, const double *points_ptr, int num_points,
                   uint32_t* results_ptr ) {
        double *polygon_data;
        double *points_data;
        uint32_t *result_data;

        auto polygon_size = num_vertices * sizeof(double) * 2;
        auto points_size = num_points * sizeof(double) * 2;
        auto results_size = num_points * sizeof(uint32_t);

        // allocate device memory for the polygon and the points to test, and copy the data over
        CUDA_CALL(cudaMalloc((void **) &polygon_data, polygon_size));
        CUDA_CALL(cudaMalloc((void **) &points_data, points_size));
        CUDA_CALL(cudaMalloc((void **) &result_data, results_size));
        CUDA_CALL(cudaMemcpy(polygon_data, poly_ptr, polygon_size, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(points_data, points_ptr, points_size, cudaMemcpyHostToDevice));

        auto block_size = BLOCK_SIZE;
        auto num_blocks = std::max(MAX_BLOCKS, int((num_points + block_size - 1) / block_size));
        check_points<<<num_blocks, block_size>>>(polygon_data, num_vertices, points_data, num_points, result_data);

        // copy the results back from the device
        CUDA_CALL(cudaMemcpy(results_ptr, result_data, results_size, cudaMemcpyDeviceToHost));

        // free everything up
        CUDA_CALL(cudaFree(polygon_data))
        CUDA_CALL(cudaFree(points_data))
        CUDA_CALL(cudaFree(result_data))
    }

    void
    visibility( const double* data, int height, int width, double* results, const int* start, const int* ends, int num_ends ) {

        double *cuda_data;
        double *cuda_result;
        int *cuda_ends;
        int *cuda_start;

        auto data_size = height * width * sizeof(double);
        auto start_size = 2 * sizeof(int);
        auto ends_size = num_ends * start_size;

        CUDA_CALL(cudaMalloc( &cuda_data, data_size));
        CUDA_CALL(cudaMemcpy( cuda_data, data, data_size, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMalloc( &cuda_start, start_size));
        CUDA_CALL(cudaMemcpy( cuda_start, start, start_size, cudaMemcpyHostToDevice));
        if( num_ends > 0 ) {
            CUDA_CALL(cudaMalloc( &cuda_ends, ends_size));
            CUDA_CALL(cudaMemcpy( cuda_ends, ends, ends_size, cudaMemcpyHostToDevice));
        } else {
            cuda_ends = nullptr;
            num_ends = height * width;
        }
        CUDA_CALL(cudaMalloc( &cuda_result, data_size));
        CUDA_CALL(cudaMemset(cuda_result, 0, data_size));

        auto block_size = BLOCK_SIZE;
        auto num_blocks = std::max(MAX_BLOCKS, int((num_ends + block_size - 1) / block_size));
        check_visibility<<<num_blocks, block_size>>>(cuda_data, height, width, cuda_start, cuda_ends, num_ends, cuda_result);

        // copy the results back from the device
        CUDA_CALL(cudaMemcpy(results, cuda_result, height * width * sizeof(double), cudaMemcpyDeviceToHost));

        // release the memory
        CUDA_CALL(cudaFree(cuda_data));
        CUDA_CALL(cudaFree(cuda_start));
        CUDA_CALL(cudaFree(cuda_result));
        if( cuda_ends != nullptr){
            CUDA_CALL(cudaFree(cuda_ends));
        }
    }


    void
    visibility_from_region( const double* data, int height, int width, double* results, const int* starts,
                            int num_starts, const int* ends, int num_ends ) {

        double *cuda_data;
        double *cuda_result;
        int *cuda_ends;
        int *cuda_start;

        auto data_size = height * width * sizeof(double);
        auto start_size = num_starts * 2 * sizeof(int);
        auto ends_size = num_ends * 2 * sizeof(int);

        CUDA_CALL(cudaMalloc( &cuda_data, data_size));
        CUDA_CALL(cudaMemcpy( cuda_data, data, data_size, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMalloc( &cuda_start, start_size));
        CUDA_CALL(cudaMemcpy( cuda_start, starts, start_size, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMalloc( &cuda_ends, ends_size));
        CUDA_CALL(cudaMemcpy( cuda_ends, ends, ends_size, cudaMemcpyHostToDevice));

        // The space required for results is now a value for each desired end-point, arranged in rows for each
        // desired starting point. NOTE: this can be very large...
        auto results_size = num_starts * num_ends * sizeof(double);
        CUDA_CALL(cudaMalloc( &cuda_result, results_size));
        CUDA_CALL(cudaMemset(cuda_result, 0, results_size));

        auto x_block_size = BLOCK_SIZE / Y_BLOCK_SIZE;
        dim3 block( x_block_size, Y_BLOCK_SIZE);
        dim3 grid( std::max( 1, std::min(MAX_BLOCKS, int((num_ends + x_block_size - 1) / x_block_size))),
                   std::max( 1, std::min(MAX_BLOCKS, int((num_starts + Y_BLOCK_SIZE - 1) / Y_BLOCK_SIZE))));

        check_region_visibility<<<grid, block>>>(cuda_data, height, width,
                                                 cuda_start, num_starts,
                                                 cuda_ends, num_ends, cuda_result);

        // copy the results back from the device
        CUDA_CALL(cudaMemcpy(results, cuda_result, results_size, cudaMemcpyDeviceToHost));

        // release the memory
        CUDA_CALL(cudaFree(cuda_data));
        CUDA_CALL(cudaFree(cuda_start));
        CUDA_CALL(cudaFree(cuda_result));
        if( cuda_ends != nullptr){
            CUDA_CALL(cudaFree(cuda_ends));
        }
   }


    void
    faux_scan( const double* polygon_list, const size_t* polygon_indices, int num_polygons, double start_x, double start_y,
               double angle_start, double angle_increment, int num_rays, double max_range, double resolution, double* results ) {

        double *cuda_data;
        size_t *cuda_index;
        double *cuda_result;

        auto poly_data_size = polygon_indices[num_polygons] * sizeof(double) * 2;
        auto poly_index_size = (num_polygons + 1) * sizeof(size_t);
        auto scan_size = num_rays * sizeof(double);

        CUDA_CALL(cudaMalloc( &cuda_data, poly_data_size));
        CUDA_CALL(cudaMemcpy( cuda_data, polygon_list, poly_data_size, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMalloc( &cuda_index, poly_index_size));
        CUDA_CALL(cudaMemcpy( cuda_index, polygon_indices, poly_index_size, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMalloc( &cuda_result, scan_size));
        CUDA_CALL(cudaMemset(cuda_result, 0, scan_size));

        auto block_size = BLOCK_SIZE;
        auto num_blocks = std::max(MAX_BLOCKS, int((num_rays + block_size - 1) / block_size));
        faux_ray<<<num_blocks, block_size>>>(cuda_data, cuda_index, num_polygons, start_x, start_y, angle_start,
                                             angle_increment, num_rays, max_range, resolution, cuda_result);

        // copy the results back from the device
        CUDA_CALL(cudaMemcpy(results, cuda_result, scan_size, cudaMemcpyDeviceToHost));

        // release the memory
        CUDA_CALL(cudaFree(cuda_data));
        CUDA_CALL(cudaFree(cuda_index));
        CUDA_CALL(cudaFree(cuda_result));
    }


}
