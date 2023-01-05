//
// Created by bjgilhul on 1/5/23.
//

#include "visibility.h"
#include <float.h>

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
    check_visibility(const double *data, const size_t height, const size_t width, const int *start,
                          const int *ends, int num_ends, double *results ) {

        auto start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (int i = start_index; i < num_ends; i += stride) {
            int ex, ey;
            if( ends == nullptr ) {
                ex = i % width;
                ey = i / width;
            } else {
                ex = ends[i*2];
                ey = ends[i*2+1];
            }
            results[ey*width + ex] = line_observation( data, height, width, start[0], start[1], ex, ey );
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

        return;
    }


}