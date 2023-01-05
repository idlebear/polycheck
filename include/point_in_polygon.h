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
#include "common.h"


namespace polycheck {

void contains( const double *poly_ptr, int num_vertices, const double *points_ptr, int num_points, uint32_t* results_ptr );

}  // polycheck


#endif // _POLYCHECK_H_
