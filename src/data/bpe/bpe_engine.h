#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// --- Terminology ---------------------------------------------------------
// token        one vocab entry.
// vocab        str to int32-ID mapping for tokens
// chunk        one pretokenized word's token sequence (vector<int32>).
// chunk_id     int32 index into tokens_by_chunk_ / chunk_counts_.
// chunk_count  number of corpus occurrences of a chunk (int64).
// pair         two adjacent token_id (a, b) inside a chunk.
// merge        replace every adjacent (a, b) in every chunk with a fresh merged token_id.
// -------------------------------------------------------------------------

// --- Internal state ------------------------------------------------------
// Private member fields owned by BpeEngine. Field names match the
// declarations in the class body below; this block documents what each
// field holds, when it is populated, and which methods may mutate it.
//
// num_threads_       OMP thread count for the parallel regions
// tokens_by_chunk_   chunk_id to vec<tokens> map
// chunk_counts_      chunk_id to its count map
// pair_counts_       pair to its count map
// chunks_by_pair_    pair to vec<chunk_id> map, which chunks contains pair 
// id2token_          token_id to token map
// token2id_          token to token_id map
// merges_            ordered merge log, stored as (a_id, b_id)
// heap_              lazy max-heap of (neg_count, pair_key)
// -------------------------------------------------------------------------

// Pack two int32 token IDs into a uint64 map key
inline uint64_t pack_pair(int32_t a, int32_t b) {
    return (
        static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
        static_cast<uint64_t>(static_cast<uint32_t>(b)
    );
}

inline int32_t unpack_a(uint64_t key) {
    return static_cast<int32_t>(static_cast<uint32_t>(key >> 32));
}

inline int32_t unpack_b(uint64_t key) {
    return static_cast<int32_t>(static_cast<uint32_t>(key & 0xFFFFFFFFu));
}

class BpeEngine {
public:
    BpeEngine() = default;
    void load_chunks(py::dict chunks, py::dict vocab);
    void set_num_threads(int n);

    std::vector<int32_t> get_chunk_tokens(int32_t chunk_id) const;
    int64_t get_chunk_count(int32_t chunk_id) const;
    int32_t get_num_chunks() const { return static_cast<int32_t>(tokens_by_chunk_.size()); }

    std::string id2token(int32_t id) const;
    int32_t token2id(const std::string& token) const;

    py::list build_pair_index();
    int64_t get_pair_count(int32_t a, int32_t b) const;
    std::vector<int32_t> get_chunks_by_pair(int32_t a, int32_t b) const;
    void drop_pair(int32_t a, int32_t b);

    py::list apply_merge(int32_t a, int32_t b, int32_t merged_id);
    void run_replay_merges(
        py::list merges,
        py::object progress_callback,
        int32_t progress_every
    );

    int run_merge_loop(
        int32_t target_vocab_size,
        py::object merge_filter,
        py::object progress_callback,
        int32_t progress_every
    );

    py::dict get_vocab() const;
    py::list get_merges() const;
    int32_t get_vocab_size() const { return static_cast<int32_t>(id2token_.size()); }

private:
    std::vector<std::vector<int32_t>> tokens_by_chunk_;
    std::vector<int64_t> chunk_counts_;
    std::unordered_map<uint64_t, int64_t> pair_counts_;
    std::unordered_map<uint64_t, std::vector<int32_t>> chunks_by_pair_;
    int num_threads_ = -1;

    std::unordered_map<uint64_t, int64_t> apply_merge_internal(
        int32_t a, int32_t b, int32_t merged_id);
    std::vector<std::string> id2token_;
    std::unordered_map<std::string, int32_t> token2id_;
    std::vector<std::pair<int32_t, int32_t>> merges_;

    struct HeapEntry {
        int64_t neg_count;
        uint64_t pair_key;
        bool operator>(const HeapEntry& other) const {
            if (neg_count != other.neg_count) return neg_count > other.neg_count;
            return pair_key > other.pair_key;
        }
    };
    std::vector<HeapEntry> heap_;
};
