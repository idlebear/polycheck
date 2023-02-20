//
// Created by bjgilhul on 1/5/23.
//

#ifndef POLYCHECK_VISIBILITY_H
#define POLYCHECK_VISIBILITY_H

#include "common.h"

namespace polycheck {

    void contains( const double *poly_ptr, int num_vertices, const double *points_ptr, int num_points, uint32_t* results_ptr );

    void visibility( const double* data, int height, int width, double* results, const int* start,
                     const int* ends = nullptr, int num_ends=0 );
    void visibility_from_region( const double* data, int height, int width, double* results, const int* starts,
                                 int num_starts, const int* ends, int num_ends );

    void faux_scan( const double* polygon_list, const size_t* polygon_indices, int num_polygons, double start_x, double start_y,
               double angle_start, double angle_increment, int num_rays, double max_range, double resolution, double *results );

}  // polycheck

#endif //POLYCHECK_VISIBILITY_H
