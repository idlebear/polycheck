//
// Created by bjgilhul on 1/5/23.
//

#ifndef POLYCHECK_COMMON_H
#define POLYCHECK_COMMON_H

namespace polycheck {

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

    // Thread allocation sizes
    const auto BLOCK_SIZE = 512;
    const auto MAX_BLOCKS = 128;

}

#endif //POLYCHECK_COMMON_H
