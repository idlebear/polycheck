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

PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<double>>);

PYBIND11_MODULE(polycheck, m) {
    py::bind_vector<std::vector<double>>(m, "VectorDouble");
    py::bind_vector<std::vector<std::vector<double>>>(m, "VectorVectorDouble");

    m.def("contains", &polycheck::contains, py::arg( "polygon"), py::arg( "points" ));
}
