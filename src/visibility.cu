//
// Created by bjgilhul on 1/5/23.
//

#include "visibility.h"
#include <float.h>
#include <iostream>

namespace polycheck {

    __device__ double
    line_observation( const double* data, int height, int width, int sx, int sy, int ex, int ey ) {
        // Using Bresenham implementation found at:
        //   http://members.chello.at/~easyfilter/Bresenham.pdf
        auto dx = abs(sx-ex);
        auto step_x = sx < ex ? 1 : -1;
        auto dy = -abs(sy-ey);
        auto step_y = sy < ey ? 1 : -1;
        auto error = dx + dy;

        auto observation = 1.0;    // assume the point is initially viewable
        for( ;; ) {
            observation *= (1.0 - data[ sy * width + sx]);
            if( observation < FLT_EPSILON*2 ) {
                break;
            }
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
        }

        return observation;
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

        std::cout << "Calling region visibility check with " << num_starts << " starts and " << num_ends << " ends." << std::endl;

        auto x_block_size = BLOCK_SIZE / Y_BLOCK_SIZE;
        dim3 block( x_block_size, Y_BLOCK_SIZE);
        dim3 grid( std::max( 1, std::min(MAX_BLOCKS, int((num_ends + x_block_size - 1) / x_block_size))),
                   std::max( 1, std::min(MAX_BLOCKS, int((num_starts + Y_BLOCK_SIZE - 1) / Y_BLOCK_SIZE))));

        std::cout << "Using a grid of size " << grid.x << "," << grid.y << " and blocks of " << block.x  << "," << block.y << std::endl;

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



}