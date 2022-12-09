//
// Created by bjgilhul on 2022-12-07.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

#include "point_in_polygon.h"

namespace py = pybind11;

// wrap C++ function with NumPy array IO
//
//   https://stackoverflow.com/questions/54793539/pybind11-modify-numpy-array-from-c
//
py::object wrapper(py::array_t<double> poly_array,
                   py::array_t<double> points_array,
                   py::array_t<uint32_t> results_array ) {
    // check input dimensions
    if ( poly_array.ndim() != 2 ) {
        throw std::runtime_error("Input should be 2-D NumPy array");
    }
    if( poly_array.shape()[1] != 2 || points_array.shape()[1] != 2 ) {
        throw std::runtime_error("Input should cartesian (X,Y) coordinates.");
    }
    if( results_array.shape()[1] != 1 ) {
        throw std::runtime_error("Output is a Points x 1 column vector of integers.");
    }

    auto poly_array_data = poly_array.request();
    auto points_array_data = points_array.request();
    auto results_array_data = results_array.request();

    auto num_vertices = poly_array.shape()[0];
    auto num_points = points_array.shape()[0];

    auto poly_ptr = (double *)poly_array_data.ptr;
    auto points_ptr = (double *)points_array_data.ptr;
    auto results_ptr = (u_int32_t *)results_array_data.ptr;

    polycheck::contains( poly_ptr, num_vertices, points_ptr, num_points, results_ptr );
    return py::cast<py::none>(Py_None);
}


PYBIND11_MODULE(polycheck, m) {
    m.def("contains", &wrapper, py::arg( "polygon"), py::arg( "points" ), py::arg( "results"));
}
