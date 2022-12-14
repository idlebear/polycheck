//
// Created by bjgilhul on 1/5/23.
//

#ifndef POLYCHECK_VISIBILITY_H
#define POLYCHECK_VISIBILITY_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "common.h"

namespace polycheck {

    void visibility( const double* data, int height, int width, double* results, const int* start,
                     const int* ends = nullptr, int num_ends=0 );
    void visibility_from_region( const double* data, int height, int width, double* results, const int* starts,
                                 int num_starts, const int* ends, int num_ends );

}  // polycheck

#endif //POLYCHECK_VISIBILITY_H
