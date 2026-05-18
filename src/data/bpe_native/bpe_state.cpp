#include "bpe_state.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

void BpeState::seed(py::dict chunks, py::dict symbol_table) {
    // 1. Resolve each chunk-tuple of str into a vector<int32_t> via symbol_table.
    //    Collect (int32_chunk, weight) pairs so we can sort them deterministically.
    std::vector<std::pair<std::vector<int32_t>, int64_t>> resolved;
    resolved.reserve(chunks.size());

    for (auto item : chunks) {
        py::tuple tup = py::reinterpret_borrow<py::tuple>(item.first);
        int64_t weight = py::cast<int64_t>(item.second);

        std::vector<int32_t> ids;
        ids.reserve(tup.size());
        for (auto sym : tup) {
            // symbol_table[sym] — raises KeyError if missing, propagating to Python.
            py::object id_obj = symbol_table[sym];
            ids.push_back(py::cast<int32_t>(id_obj));
        }
        resolved.emplace_back(std::move(ids), weight);
    }

    // 2. Sort by chunk-tuple of int32 IDs — same order as Python's
    //    sorted(chunks.items(), key=lambda kv: kv[0]) over str-tuples, because
    //    the byte alphabet's IDs are assigned in lex-of-unicode-char order
    //    and chunks at seed time contain only byte-alphabet single-char tokens.
    std::sort(resolved.begin(), resolved.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.first < rhs.first;
              });

    // 3. Move into final storage.
    symbols_per_chunk_.clear();
    chunk_weights_.clear();
    symbols_per_chunk_.reserve(resolved.size());
    chunk_weights_.reserve(resolved.size());
    for (auto& [ids, weight] : resolved) {
        symbols_per_chunk_.push_back(std::move(ids));
        chunk_weights_.push_back(weight);
    }
}

std::vector<int32_t> BpeState::get_chunk_symbols(int32_t chunk_id) const {
    if (chunk_id < 0 || chunk_id >= static_cast<int32_t>(symbols_per_chunk_.size())) {
        throw std::out_of_range("chunk_id out of range");
    }
    return symbols_per_chunk_[chunk_id];
}

int64_t BpeState::get_chunk_weight(int32_t chunk_id) const {
    if (chunk_id < 0 || chunk_id >= static_cast<int32_t>(chunk_weights_.size())) {
        throw std::out_of_range("chunk_id out of range");
    }
    return chunk_weights_[chunk_id];
}
