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
// chunk_id     int32 index into chunks_.
// chunk_count  number of corpus occurrences of a chunk (int64).
// pair         two adjacent token_id (a, b) inside a chunk.
// merge        replace every adjacent (a, b) in every chunk with a fresh merged token_id.
// -------------------------------------------------------------------------

// --- Internal state ------------------------------------------------------
// Private member fields owned by BpeEngine.
//
// num_threads_   OMP thread count for the parallel regions
// chunks_        chunk_id -> Chunk { id, tokens, count }
// pairs_         packed pair_key -> Pair { a, b, count, chunk_ids }
// id2token_      token_id to token map
// token2id_      token to token_id map
// merges_        ordered merge log, stored as (a_id, b_id)
// heap_          lazy max-heap of (neg_count, pair_key)
// -------------------------------------------------------------------------

struct Chunk {
    int32_t id;
    std::vector<int32_t> tokens;
    int64_t count;
};

struct Pair {
    int32_t a;
    int32_t b;
    int64_t count;
    // Invariant: no duplicates. Maintainers must dedupe at every insertion
    // site (init_pairs_, merge_one_chunk_'s delta emission) since vector
    // doesn't enforce it. Vector chosen over unordered_set for ~8x lower
    // per-element memory.
    std::vector<int32_t> chunk_ids;

    // Pack (a, b) into a uint64 hash-map key. High 32 bits = a, low 32 = b.
    static uint64_t pack(int32_t a, int32_t b);
    // Recover (a, b) from a packed key.
    static std::pair<int32_t, int32_t> unpack(uint64_t key);
};

// Per-thread reduction buffer for the parallel apply_merge region.
// Direct writes to the shared pairs_ map would race on bucket structure;
// each thread accumulates locally and the main thread merges after the
// parallel section.
struct ThreadLocalDelta {
    std::vector<std::pair<uint64_t, int64_t>> pair_count_deltas;
    std::vector<std::pair<uint64_t, int32_t>> chunk_id_adds;

    void add_pair_delta(uint64_t pair_key, int64_t delta);
    void add_chunk_delta(uint64_t pair_key, int32_t chunk_id);
};

class BpeEngine {
public:
    BpeEngine() = default;
    void set_num_threads(int n);

    // Load the corpus and build the initial pair index in one shot. After
    // feed() returns, the engine is ready for either train() (fresh) or
    // replay_merges() then train() (resume).
    void feed(py::dict chunks, py::dict vocab);

    // Replay a pre-computed merge list and refresh the pair index.
    void replay_merges(
        py::list merges,
        py::object progress_callback,
        int32_t progress_every
    );

    // Train new merges by greedy max-frequency selection until vocab reaches
    // target_vocab_size (or no more eligible pairs). Returns number of merges
    // accepted this call.
    int train(
        int32_t target_vocab_size,
        py::object merge_filter,
        py::object progress_callback,
        int32_t progress_every
    );

    // Read-only inspection helpers (used by tests + trainer).
    std::vector<int32_t> get_chunk_tokens(int32_t chunk_id) const;
    int64_t get_chunk_count(int32_t chunk_id) const;
    int32_t get_num_chunks() const;

    std::string id2token(int32_t id) const;
    int32_t token2id(const std::string& token) const;

    py::list list_pairs() const;  // [(a, b, count), ...] snapshot
    int64_t get_pair_count(int32_t a, int32_t b) const;
    std::vector<int32_t> get_chunks_by_pair(int32_t a, int32_t b) const;
    void drop_pair(int32_t a, int32_t b);

    py::list apply_merge(int32_t a, int32_t b, int32_t merged_id);

    py::dict get_vocab() const;
    py::list get_merges() const;
    int32_t get_vocab_size() const;

private:
    // Internal lifecycle steps; not exposed to Python. feed() and
    // replay_merges() compose these in the correct order.
    void init_chunks_(py::dict chunks, py::dict vocab);
    void init_pairs_();
    void replay_merges_(
        py::list merges,
        py::object progress_callback,
        int32_t progress_every
    );

    std::unordered_map<uint64_t, int64_t> apply_merge_(
        int32_t a, int32_t b, int32_t merged_id);

    // Left-to-right collapse of every (a, b) into `merged` inside one chunk.
    // If `out` is non-null, also emits the pair-count and chunk-membership
    // deltas induced by the merge (used during training); pass nullptr for
    // pure replay.
    static bool merge_one_chunk_(
        Chunk& chunk,
        int32_t a,
        int32_t b,
        int32_t merged,
        ThreadLocalDelta* out = nullptr
    );

    int num_threads_ = -1;
    std::vector<Chunk> chunks_;
    std::unordered_map<uint64_t, Pair> pairs_;

    std::vector<std::string> id2token_;
    std::unordered_map<std::string, int32_t> token2id_;
    std::vector<std::pair<int32_t, int32_t>> merges_;

    struct HeapEntry {
        int64_t neg_count;
        uint64_t pair_key;
        bool operator>(const HeapEntry& other) const;
    };
    std::vector<HeapEntry> heap_;
};
