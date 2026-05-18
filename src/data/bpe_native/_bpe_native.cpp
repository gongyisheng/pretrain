#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

static std::string hello() {
    return "bpe_native loaded";
}

PYBIND11_MODULE(_bpe_native, m) {
    m.doc() = "Native C++ hot loops for BPE training (see src/data/bpe.py).";
    m.def("hello", &hello, "Smoke-test that the extension built and loads.");
}
