// Copyright (c) 2022 Barry Gilhuly and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef _POLYCHECK_H_
#define _POLYCHECK_H_

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <memory>
#include <vector>
#include <stdio.h>


#ifdef DEBUG

#ifndef CUDA_CALL
#define CUDA_CALL(call)\
{\
    auto status = static_cast<cudaError_t>(call);\
    if (status != cudaSuccess) {\
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n",\
            #call, __LINE__, __FILE__, cudaGetErrorString(status), status);\
    }\
}
#endif

#else

#ifndef CUDA_CALL
#define CUDA_CALL(call)\
{\
    auto status = static_cast<cudaError_t>(call);\
    assert( status == cudaSuccess);\
}
#endif

#endif

namespace polycheck {

const auto BLOCK_SIZE = 256;
const auto MAX_BLOCKS = 64;

void contains( const double *poly_ptr, int num_vertices, const double *points_ptr, int num_points, uint32_t* results_ptr );

}  // polycheck


#endif // _POLYCHECK_H_
