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
        .def("num_chunks", &BpeState::num_chunks)
        .def("build_initial_pairs", &BpeState::build_initial_pairs)
        .def("pair_count", &BpeState::pair_count, py::arg("a"), py::arg("b"))
        .def("pair_chunks", &BpeState::pair_chunks, py::arg("a"), py::arg("b"))
        .def("replay_merges", &BpeState::replay_merges, py::arg("merges"))
        .def("apply_merge", &BpeState::apply_merge,
             py::arg("a"), py::arg("b"), py::arg("merged_id"))
        .def("drop_pair", &BpeState::drop_pair, py::arg("a"), py::arg("b"))
        .def("set_num_threads", &BpeState::set_num_threads, py::arg("n"));
}
