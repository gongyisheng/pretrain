#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bpe_engine.h"

namespace py = pybind11;

static std::string hello() {
    return "bpe_native loaded";
}

PYBIND11_MODULE(_bpe_engine, m) {
    m.doc() = "Native C++ hot loops for BPE training (see src/data/bpe.py).";
    m.def("hello", &hello, "Smoke-test that the extension built and loads.");

    py::class_<BpeEngine>(m, "BpeEngine")
        .def(py::init<>())
        .def("seed", &BpeEngine::seed, py::arg("chunks"), py::arg("symbol_table"))
        .def("set_num_threads", &BpeEngine::set_num_threads, py::arg("n"))
        .def("get_chunk_symbols", &BpeEngine::get_chunk_symbols, py::arg("chunk_id"))
        .def("get_chunk_weight", &BpeEngine::get_chunk_weight, py::arg("chunk_id"))
        .def("get_num_chunks", &BpeEngine::get_num_chunks)
        .def("id2sym", &BpeEngine::id2sym, py::arg("id"))
        .def("sym2id", &BpeEngine::sym2id, py::arg("sym"))
        .def("build_initial_pairs", &BpeEngine::build_initial_pairs)
        .def("pair_count", &BpeEngine::pair_count, py::arg("a"), py::arg("b"))
        .def("pair_chunks", &BpeEngine::pair_chunks, py::arg("a"), py::arg("b"))
        .def("drop_pair", &BpeEngine::drop_pair, py::arg("a"), py::arg("b"))
        .def("apply_merge", &BpeEngine::apply_merge,
             py::arg("a"), py::arg("b"), py::arg("merged_id"))
        .def("run_replay_merges", &BpeEngine::run_replay_merges,
             py::arg("merges"),
             py::arg("progress_callback") = py::none(),
             py::arg("progress_every") = 1000,
             py::arg("show_progress") = false,
             py::arg("progress_desc") = std::string(""))
        .def("run_merge_loop", &BpeEngine::run_merge_loop,
             py::arg("target_vocab_size"),
             py::arg("merge_filter") = py::none(),
             py::arg("progress_callback") = py::none(),
             py::arg("progress_every") = 1000,
             py::arg("show_progress") = false,
             py::arg("progress_desc") = std::string(""))
        .def("get_vocab", &BpeEngine::get_vocab)
        .def("get_merges", &BpeEngine::get_merges)
        .def("get_vocab_size", &BpeEngine::get_vocab_size);
}
