#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>
#include <string>
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

    // Apply a list of merges in order to every chunk, mutating in place.
    // merges: list[(a_id, b_id, merged_id)]. Parallel across chunks; each
    // thread runs the full merge list per chunk. Must NOT be called after
    // build_initial_pairs (replay happens before pair-counting).
    void replay_merges(py::list merges);

    // Apply merge (a, b) → merged_id everywhere it currently appears.
    // Mutates symbols_per_chunk in place, updates pair_counts and where_
    // to reflect neighbor-pair changes, and returns a list of
    // (pair_a, pair_b, delta_count) for every pair whose count changed
    // (excluding the merged pair itself, which is removed). GIL released
    // during the parallel chunk scan. Returns the deltas list.
    py::list apply_merge(int32_t a, int32_t b, int32_t merged_id);

    // Veto-path helper: drop pair from pair_counts_ and where_ so it's
    // never considered again. Used by Python when merge_filter rejects.
    void drop_pair(int32_t a, int32_t b);

    // Configure OpenMP thread count. -1 means use omp_get_max_threads().
    void set_num_threads(int n);

    // SuperBPE filter configuration. Both default to "disabled."
    // max_superword_words: reject a merge if the merged token's word-count
    //   (counted by 'Ġ' chars + 1 if the merged str does not start with 'Ġ')
    //   exceeds this. -1 disables the check.
    // forbid_colon_g: when true, reject any merge whose merged string
    //   contains the substring ":Ġ".
    // The two filters are independent and BOTH are evaluated when set.
    void set_max_superword_words(int n);
    void set_forbid_colon_g(bool flag);
    int get_max_superword_words() const { return max_superword_words_; }
    bool get_forbid_colon_g() const { return forbid_colon_g_; }

    // Run the merge loop from current state (call sequence: seed →
    // optional replay_merges → build_initial_pairs → THIS). Initializes
    // the native heap from pair_counts_ on entry. Returns the number of
    // accepted merges. The final (vocab, merges) is fetched via
    // get_vocab() / get_merges() after the loop returns.
    //
    // progress_cb (may be None) is called every progress_every accepted
    // merges with (current_vocab_size, vocab_snapshot, merges_snapshot).
    // show_progress toggles a tqdm-style progress bar to stderr.
    int run_merge_loop(int32_t target_vocab_size,
                       py::object progress_cb,
                       int32_t progress_every,
                       bool show_progress,
                       const std::string& progress_desc);

    // Marshal the final native state back to Python.
    py::dict get_vocab() const;          // dict[str, int]
    py::list get_merges() const;          // list[tuple[str, str]]

    // Test/debug accessors.
    std::vector<int32_t> get_chunk_symbols(int32_t chunk_id) const;
    int64_t get_chunk_weight(int32_t chunk_id) const;
    int32_t num_chunks() const { return static_cast<int32_t>(symbols_per_chunk_.size()); }
    int64_t pair_count(int32_t a, int32_t b) const;
    std::vector<int32_t> pair_chunks(int32_t a, int32_t b) const;

    // v2 test/debug accessors: expose the native vocab for verification.
    std::string id2sym(int32_t id) const;
    int32_t vocab_id(const std::string& sym) const;
    int32_t native_vocab_size() const { return static_cast<int32_t>(id2sym_native_.size()); }

private:
    std::vector<std::vector<int32_t>> symbols_per_chunk_;
    std::vector<int64_t> chunk_weights_;
    // (pair, count) and (pair → chunk_ids). uint64 key = pack_pair(a, b).
    std::unordered_map<uint64_t, int64_t> pair_counts_;
    std::unordered_map<uint64_t, std::vector<int32_t>> where_;
    int num_threads_ = -1;  // -1 = use omp default

    // Core merge logic, exposed for both the public apply_merge (which
    // wraps it in a py::list construction) and run_merge_loop (which
    // consumes the deltas directly to avoid the boundary cost). Updates
    // symbols_per_chunk_, pair_counts_, and where_ in place; returns the
    // per-pair delta map (one entry per pair whose count changed,
    // excluding the merged pair itself).
    std::unordered_map<uint64_t, int64_t> apply_merge_internal(
        int32_t a, int32_t b, int32_t merged_id);
    // SuperBPE filter config — read by run_merge_loop (v2 Task 4).
    int max_superword_words_ = -1;   // -1 = disabled
    bool forbid_colon_g_ = false;
    // String ↔ ID vocab, owned natively so run_merge_loop can build merged
    // token strings and evaluate the SuperBPE filter without GIL acquires.
    // Populated by seed() (and grown by run_merge_loop in v2 Task 4).
    std::vector<std::string> id2sym_native_;
    std::unordered_map<std::string, int32_t> vocab_native_;

    // Accepted merges in order; (a_id, b_id). The merged token string is
    // id2sym_native_[a_id] + id2sym_native_[b_id], registered in
    // vocab_native_ on accept.
    std::vector<std::pair<int32_t, int32_t>> merges_native_;

    // Native heap of packed (neg_count, pair_key) entries; built lazily
    // by run_merge_loop from pair_counts_. The std::greater operator>
    // sorts smaller neg_count (= larger positive count) first; ties
    // break ascending on pair_key, matching HF's (id_a, id_b) tie-break.
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
