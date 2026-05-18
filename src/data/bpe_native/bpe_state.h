#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// Pack two int32 symbol IDs into a single uint64 key for unordered_map.
// We use this everywhere we'd otherwise need std::pair<int32_t,int32_t>
// as a map key — uint64 avoids a custom hash and packs into one register.
inline uint64_t pack_pair(int32_t a, int32_t b) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(b));
}

inline int32_t unpack_a(uint64_t key) {
    return static_cast<int32_t>(static_cast<uint32_t>(key >> 32));
}

inline int32_t unpack_b(uint64_t key) {
    return static_cast<int32_t>(static_cast<uint32_t>(key & 0xFFFFFFFFu));
}

class BpeState {
public:
    BpeState() = default;

    // Convert a Python chunks dict (dict[tuple[str], int]) into int32 arrays.
    // Sorts by chunk-tuple of int32 IDs for deterministic chunk_id assignment.
    // symbol_table: Python's vocab at seed time (dict[str, int32]) — must
    // contain every str symbol that appears in any chunk tuple.
    void seed(py::dict chunks, py::dict symbol_table);

    // Count adjacent-pair occurrences (weighted by chunk weight) and build
    // the pair → chunk_ids reverse index. Must be called once after seed
    // (and after replay_merges if resuming).
    // Returns list[(a_id, b_id, count)] for the Python heap.
    py::list build_initial_pairs();

    // Test/debug accessors.
    std::vector<int32_t> get_chunk_symbols(int32_t chunk_id) const;
    int64_t get_chunk_weight(int32_t chunk_id) const;
    int32_t num_chunks() const { return static_cast<int32_t>(symbols_per_chunk_.size()); }
    int64_t pair_count(int32_t a, int32_t b) const;
    std::vector<int32_t> pair_chunks(int32_t a, int32_t b) const;

private:
    std::vector<std::vector<int32_t>> symbols_per_chunk_;
    std::vector<int64_t> chunk_weights_;
    // (pair, count) and (pair → chunk_ids). uint64 key = pack_pair(a, b).
    std::unordered_map<uint64_t, int64_t> pair_counts_;
    std::unordered_map<uint64_t, std::vector<int32_t>> where_;
};
