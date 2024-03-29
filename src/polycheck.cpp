//
// Created by bjgilhul on 2022-12-07.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <utility>
#include <vector>

#include "visibility.h"

namespace py = pybind11;

// wrap C++ function with NumPy array IO
//
//   https://stackoverflow.com/questions/54793539/pybind11-modify-numpy-array-from-c
//
py::object contain_wrapper(py::array_t<double> poly_array,
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

    auto num_vertices = int(poly_array.shape()[0]);
    auto num_points = int(points_array.shape()[0]);

    auto poly_ptr = (double *)poly_array_data.ptr;
    auto points_ptr = (double *)points_array_data.ptr;
    auto results_ptr = (u_int32_t *)results_array_data.ptr;

    polycheck::contains( poly_ptr, num_vertices, points_ptr, num_points, results_ptr );
    return py::cast<py::none>(Py_None);
}


py::object visibility_wrapper(py::array_t<double> grid_array,
                              py::array_t<int> start_array,
                              py::array_t<double> results_array ) {
    // check input dimensions
    if ( grid_array.ndim() != 2 ) {
        throw std::runtime_error("Input should be 2-D NumPy array");
    }

    auto height = int(grid_array.shape()[0]);
    auto width = int(grid_array.shape()[1]);
    if( start_array.shape()[0] != 1 || start_array.shape()[1] != 2 ) {
        throw std::runtime_error("Input should cartesian (X,Y) coordinates.");
    }
    if( results_array.shape()[0] != height || results_array.shape()[1] != width ) {
        throw std::runtime_error("Output must be the same size as the input array.");
    }

    auto grid_array_data = grid_array.request();
    auto start_array_data = start_array.request();
    auto results_array_data = results_array.request();

    auto data_ptr = (double *)grid_array_data.ptr;
    auto start_ptr = (int *)start_array_data.ptr;
    auto results_ptr = (double *)results_array_data.ptr;

    polycheck::visibility( data_ptr, height, width, results_ptr, start_ptr );
    return py::cast<py::none>(Py_None);
}


py::object region_visibility_wrapper(py::array_t<double> grid_array,
                              py::array_t<int> start_array,
                              py::array_t<int> ends_array,
                              py::array_t<double> results_array ) {
    // check input dimensions
    if ( grid_array.ndim() != 2 ) {
        throw std::runtime_error("Input should be 2-D NumPy array");
    }

    auto height = int(grid_array.shape()[0]);
    auto width = int(grid_array.shape()[1]);
    if( start_array.shape()[0] != 1 || start_array.shape()[1] != 2 ) {
        throw std::runtime_error("Input should cartesian (X,Y) coordinates.");
    }
    if( results_array.shape()[0] != height || results_array.shape()[1] != width ) {
        throw std::runtime_error("Output must be the same size as the input array.");
    }

    auto grid_array_data = grid_array.request();
    auto start_array_data = start_array.request();
    auto ends_array_data = ends_array.request();
    auto results_array_data = results_array.request();

    auto data_ptr = (double *)grid_array_data.ptr;
    auto start_ptr = (int *)start_array_data.ptr;
    auto ends_ptr = (int *)ends_array_data.ptr;
    auto results_ptr = (double *)results_array_data.ptr;

    polycheck::visibility( data_ptr, height, width, results_ptr, start_ptr, ends_ptr, int(ends_array.shape()[0]) );
    return py::cast<py::none>(Py_None);
}



py::object visibility_from_region_wrapper(py::array_t<double> grid_array,
                                          py::array_t<int> starts_array,
                                          py::array_t<int> ends_array,
                                          py::array_t<double> results_array ) {
    // check input dimensions
    if ( grid_array.ndim() != 2 ) {
        throw std::runtime_error("Input should be 2-D NumPy array");
    }

    auto height = int(grid_array.shape()[0]);
    auto width = int(grid_array.shape()[1]);
    if( starts_array.shape()[1] != 2 ) {
        throw std::runtime_error("Start location should cartesian (X,Y) coordinates.");
    }
    if( ends_array.shape()[1] != 2 ) {
        throw std::runtime_error("End locations should cartesian (X,Y) coordinates.");
    }
    if( results_array.shape()[0] != starts_array.shape()[0] ||
        results_array.shape()[1] != ends_array.shape()[0] ) {
        throw std::runtime_error("Output must be N_starts x N_ends array.");
    }

    auto grid_array_data = grid_array.request();
    auto start_array_data = starts_array.request();
    auto ends_array_data = ends_array.request();
    auto results_array_data = results_array.request();

    auto data_ptr = (double *)grid_array_data.ptr;
    auto start_ptr = (int *)start_array_data.ptr;
    auto ends_ptr = (int *)ends_array_data.ptr;
    auto results_ptr = (double *)results_array_data.ptr;

    polycheck::visibility_from_region( data_ptr, height, width, results_ptr, start_ptr, int(starts_array.shape()[0]),
                                       ends_ptr, int(ends_array.shape()[0]) );
    return py::cast<py::none>(Py_None);
}

PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<double>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::vector<double>>>);

py::object faux_scan_wrapper(std::vector<std::vector<std::vector<double>>> polygon_list,
                             double start_x, double start_y, double start_angle, double angle_increment,
                             int num_rays, double max_range, double resolution,
                             py::array_t<double> results_array ) {

    size_t num_polys = polygon_list.size();
    size_t num_vertices = 0;

    auto polygon_data = std::vector<double>();
    auto polygon_vertices = std::vector<size_t>();

    size_t current_vertex_index = 0;
    polygon_vertices.emplace_back( current_vertex_index );

    auto poly_count = 0;
    for (const auto poly: polygon_list) {
        for (const auto point: poly) {
            polygon_data.emplace_back(point[0]);
            polygon_data.emplace_back(point[1]);
        }
        current_vertex_index += poly.size();
        polygon_vertices.emplace_back( current_vertex_index );
    };

    auto results_array_data = results_array.request();
    auto results_ptr = (double *)results_array_data.ptr;

    polycheck::faux_scan( polygon_data.data(), polygon_vertices.data(), polygon_list.size(), start_x, start_y, start_angle, angle_increment, num_rays, max_range, resolution, results_ptr);

    return py::cast<py::none>(Py_None);
}


PYBIND11_MODULE(polycheck, m) {
    py::bind_vector<std::vector<double>>(m, "Vertex");
    py::bind_vector<std::vector<std::vector<double>>>(m, "VertexList");
    py::bind_vector<std::vector<std::vector<std::vector<double>>>>(m, "PolygonList");

    m.def("contains", &contain_wrapper, py::arg( "polygon"), py::arg( "points" ), py::arg( "results"));
    m.def("visibility", &visibility_wrapper, py::arg( "grid"), py::arg( "start" ), py::arg( "results"));
    m.def("region_visibility", &region_visibility_wrapper, py::arg( "grid"), py::arg( "start" ),  py::arg( "ends" ), py::arg( "results"));
    m.def("visibility_from_region", &visibility_from_region_wrapper, py::arg( "grid"), py::arg( "starts" ),  py::arg( "ends" ), py::arg( "results"));
    m.def("faux_scan", &faux_scan_wrapper, py::arg( "polygons"), py::arg( "start_x" ), py::arg( "start_y" ), py::arg( "start_angle" ), py::arg( "angle_increment" ),
          py::arg("num_rays"), py::arg( "max_range" ), py::arg( "resolution" ), py::arg( "results"));
}


