#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bpe_state.h"

namespace py = pybind11;

static std::string hello() {
    return "bpe_native loaded";
}

PYBIND11_MODULE(_bpe_native, m) {
    m.doc() = "Native C++ hot loops for BPE training (see src/data/bpe.py).";
    m.def("hello", &hello, "Smoke-test that the extension built and loads.");

    py::class_<BpeState>(m, "BpeState")
        .def(py::init<>())
        .def("seed", &BpeState::seed, py::arg("chunks"), py::arg("symbol_table"))
        .def("get_chunk_symbols", &BpeState::get_chunk_symbols, py::arg("chunk_id"))
        .def("get_chunk_weight", &BpeState::get_chunk_weight, py::arg("chunk_id"))
        .def("num_chunks", &BpeState::num_chunks);
}
