#include "bpe_state.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <omp.h>

namespace py = pybind11;

namespace {

// Single-chunk left-to-right merge. Matches the Python `_replay_merges`
// inner loop byte-for-byte: collapse adjacent (a,b) → merged from left,
// `i` stays put after a collapse (so overlapping pairs like a b a b a b
// chain correctly).
inline void replay_one_chunk(std::vector<int32_t>& syms,
                             const std::vector<std::tuple<int32_t, int32_t, int32_t>>& merges) {
    for (const auto& [a, b, merged] : merges) {
        if (syms.size() < 2) continue;
        size_t i = 0;
        while (i + 1 < syms.size()) {
            if (syms[i] == a && syms[i + 1] == b) {
                syms[i] = merged;
                syms.erase(syms.begin() + static_cast<long>(i) + 1);
                // i stays put so we can keep collapsing overlaps to the right.
            } else {
                ++i;
            }
        }
    }
}

}  // namespace

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

py::list BpeState::build_initial_pairs() {
    pair_counts_.clear();
    where_.clear();

    const int32_t n = static_cast<int32_t>(symbols_per_chunk_.size());
    for (int32_t cid = 0; cid < n; ++cid) {
        const auto& syms = symbols_per_chunk_[cid];
        const int64_t w = chunk_weights_[cid];
        if (syms.size() < 2) continue;
        // Track which pairs this chunk contributes to, to avoid duplicate
        // entries in where_[pair] for chunks where the same pair occurs
        // multiple times.
        std::unordered_set<uint64_t> seen_in_chunk;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            uint64_t key = pack_pair(syms[i], syms[i + 1]);
            pair_counts_[key] += w;
            if (seen_in_chunk.insert(key).second) {
                where_[key].push_back(cid);
            }
        }
    }

    py::list out;
    for (const auto& [key, count] : pair_counts_) {
        out.append(py::make_tuple(unpack_a(key), unpack_b(key), count));
    }
    return out;
}

int64_t BpeState::pair_count(int32_t a, int32_t b) const {
    auto it = pair_counts_.find(pack_pair(a, b));
    if (it == pair_counts_.end()) return 0;
    return it->second;
}

std::vector<int32_t> BpeState::pair_chunks(int32_t a, int32_t b) const {
    auto it = where_.find(pack_pair(a, b));
    if (it == where_.end()) return {};
    return it->second;
}

void BpeState::set_num_threads(int n) {
    num_threads_ = n;
}

void BpeState::replay_merges(py::list merges) {
    // Convert merges to a flat C++ vector once, before releasing the GIL.
    std::vector<std::tuple<int32_t, int32_t, int32_t>> ms;
    ms.reserve(merges.size());
    for (auto item : merges) {
        py::tuple t = py::reinterpret_borrow<py::tuple>(item);
        ms.emplace_back(py::cast<int32_t>(t[0]),
                        py::cast<int32_t>(t[1]),
                        py::cast<int32_t>(t[2]));
    }

    const int n = static_cast<int>(symbols_per_chunk_.size());
    if (num_threads_ > 0) {
        omp_set_num_threads(num_threads_);
    }

    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(dynamic, 64)
        for (int cid = 0; cid < n; ++cid) {
            replay_one_chunk(symbols_per_chunk_[cid], ms);
        }
    }
}
