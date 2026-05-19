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
        .def("load_chunks", &BpeEngine::load_chunks, py::arg("chunks"), py::arg("vocab"))
        .def("set_num_threads", &BpeEngine::set_num_threads, py::arg("n"))
        .def("get_chunk_tokens", &BpeEngine::get_chunk_tokens, py::arg("chunk_id"))
        .def("get_chunk_count", &BpeEngine::get_chunk_count, py::arg("chunk_id"))
        .def("get_num_chunks", &BpeEngine::get_num_chunks)
        .def("id2token", &BpeEngine::id2token, py::arg("id"))
        .def("token2id", &BpeEngine::token2id, py::arg("token"))
        .def("build_pair_index", &BpeEngine::build_pair_index)
        .def("get_pair_count", &BpeEngine::get_pair_count, py::arg("a"), py::arg("b"))
        .def("get_chunks_by_pair", &BpeEngine::get_chunks_by_pair, py::arg("a"), py::arg("b"))
        .def("drop_pair", &BpeEngine::drop_pair, py::arg("a"), py::arg("b"))
        .def("apply_merge", &BpeEngine::apply_merge,
             py::arg("a"), py::arg("b"), py::arg("merged_id"))
        .def("run_replay_merges", &BpeEngine::run_replay_merges,
             py::arg("merges"),
             py::arg("progress_callback") = py::none(),
             py::arg("progress_every") = 1000)
        .def("run_merge_loop", &BpeEngine::run_merge_loop,
             py::arg("target_vocab_size"),
             py::arg("merge_filter") = py::none(),
             py::arg("progress_callback") = py::none(),
             py::arg("progress_every") = 1000)
        .def("get_vocab", &BpeEngine::get_vocab)
        .def("get_merges", &BpeEngine::get_merges)
        .def("get_vocab_size", &BpeEngine::get_vocab_size);
}
