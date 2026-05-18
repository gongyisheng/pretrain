#include "bpe_engine.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <omp.h>

namespace py = pybind11;

namespace {

// Single-chunk, single-merge left-to-right collapse. Matches the Python
// `_replay_merges` inner loop byte-for-byte. After a collapse we advance
// `i` by 1: safe because `merged` is a fresh symbol id and can't equal
// `a`, so the next iteration would not match at the same `i` anyway.
inline void replay_single_merge_one_chunk(std::vector<int32_t>& syms,
                                          int32_t a, int32_t b, int32_t merged) {
    if (syms.size() < 2) return;
    size_t i = 0;
    while (i + 1 < syms.size()) {
        if (syms[i] == a && syms[i + 1] == b) {
            syms[i] = merged;
            syms.erase(syms.begin() + static_cast<long>(i) + 1);
            ++i;
        } else {
            ++i;
        }
    }
}

// Per-chunk scan for `apply_merge`. Mutates `syms` in place to collapse
// every (a,b) → merged. Emits delta updates for the four neighbor-pair
// patterns (prev,a)↓ (prev,merged)↑ (b,nxt)↓ (merged,nxt)↑, each scaled
// by chunk weight `w`. Records the chunk_id under any NEW pair so the
// reverse index `where_` stays current.
//
// Returns `true` if the chunk was mutated.
//
// Mirrors `_apply_merge_on_chunks` in src/data/bpe.py: left-to-right
// linear scan; after a collapse `i` stays put so overlapping patterns
// chain correctly (same overlap rule as `replay_one_chunk`).
struct LocalDeltas {
    std::vector<std::pair<uint64_t, int64_t>> count_deltas;  // (packed pair, dv)
    std::vector<std::pair<uint64_t, int32_t>> where_adds;    // (packed pair, chunk_id)
};

inline bool apply_merge_one_chunk(std::vector<int32_t>& syms,
                                  int32_t chunk_id,
                                  int64_t w,
                                  int32_t a,
                                  int32_t b,
                                  int32_t merged,
                                  LocalDeltas& out) {
    if (syms.size() < 2) return false;
    // Track new (prev,merged) / (merged,nxt) pair keys this chunk has
    // already added to where_adds — mirrors the set-based dedupe in
    // Python's `_apply_merge_on_chunks` (`wtu_adds.setdefault(p, set()).add(chunk_id)`).
    // Without it, the same chunk_id can land in where_[pair] multiple times,
    // leading to a data race on the next apply_merge for that pair.
    std::unordered_set<uint64_t> seen_in_chunk;
    bool changed = false;
    size_t i = 0;
    while (i + 1 < syms.size()) {
        if (syms[i] != a || syms[i + 1] != b) {
            ++i;
            continue;
        }
        changed = true;
        if (i > 0) {
            int32_t prev = syms[i - 1];
            out.count_deltas.emplace_back(pack_pair(prev, a), -w);
            uint64_t k_pm = pack_pair(prev, merged);
            out.count_deltas.emplace_back(k_pm, w);
            if (seen_in_chunk.insert(k_pm).second) {
                out.where_adds.emplace_back(k_pm, chunk_id);
            }
        }
        if (i + 2 < syms.size()) {
            int32_t nxt = syms[i + 2];
            out.count_deltas.emplace_back(pack_pair(b, nxt), -w);
            uint64_t k_mn = pack_pair(merged, nxt);
            out.count_deltas.emplace_back(k_mn, w);
            if (seen_in_chunk.insert(k_mn).second) {
                out.where_adds.emplace_back(k_mn, chunk_id);
            }
        }
        syms[i] = merged;
        syms.erase(syms.begin() + static_cast<long>(i) + 1);
        ++i;  // safe to advance: syms[i] is now `merged` which can't equal `a`
              // (merged is a fresh symbol id), so the next iteration would
              // not match anyway. Advancing by 1 skips the redundant check.
    }
    return changed;
}

}  // namespace

void BpeEngine::seed(py::dict chunks, py::dict symbol_table) {
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

    // Phase 4: record the symbol-table contents natively so run_merge_loop
    // can build merged-token strings without bouncing through Python.
    // We use the int32 IDs as indices into id2sym_native_, so resize first
    // then write each (str, id) into the right slot.
    id2sym_native_.clear();
    vocab_native_.clear();
    int32_t max_id = -1;
    for (auto item : symbol_table) {
        int32_t id = py::cast<int32_t>(item.second);
        if (id > max_id) max_id = id;
    }
    id2sym_native_.assign(max_id + 1, std::string{});
    for (auto item : symbol_table) {
        std::string sym = py::cast<std::string>(item.first);
        int32_t id = py::cast<int32_t>(item.second);
        id2sym_native_[id] = sym;
        vocab_native_[sym] = id;
    }
}

std::vector<int32_t> BpeEngine::get_chunk_symbols(int32_t chunk_id) const {
    if (chunk_id < 0 || chunk_id >= static_cast<int32_t>(symbols_per_chunk_.size())) {
        throw std::out_of_range("chunk_id out of range");
    }
    return symbols_per_chunk_[chunk_id];
}

int64_t BpeEngine::get_chunk_weight(int32_t chunk_id) const {
    if (chunk_id < 0 || chunk_id >= static_cast<int32_t>(chunk_weights_.size())) {
        throw std::out_of_range("chunk_id out of range");
    }
    return chunk_weights_[chunk_id];
}

py::list BpeEngine::build_initial_pairs() {
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

int64_t BpeEngine::pair_count(int32_t a, int32_t b) const {
    auto it = pair_counts_.find(pack_pair(a, b));
    if (it == pair_counts_.end()) return 0;
    return it->second;
}

std::vector<int32_t> BpeEngine::pair_chunks(int32_t a, int32_t b) const {
    auto it = where_.find(pack_pair(a, b));
    if (it == where_.end()) return {};
    return it->second;
}

void BpeEngine::set_num_threads(int n) {
    num_threads_ = n;
}

void BpeEngine::run_replay_merges(py::list merges,
                                  py::object progress_callback,
                                  int32_t progress_every) {
    // Convert merges to a flat C++ vector once.
    std::vector<std::tuple<int32_t, int32_t, int32_t>> ms;
    ms.reserve(merges.size());
    for (auto item : merges) {
        py::tuple t = py::reinterpret_borrow<py::tuple>(item);
        ms.emplace_back(py::cast<int32_t>(t[0]),
                        py::cast<int32_t>(t[1]),
                        py::cast<int32_t>(t[2]));
    }

    const int n_chunks = static_cast<int>(symbols_per_chunk_.size());
    const int n_merges = static_cast<int>(ms.size());
    if (num_threads_ > 0) {
        omp_set_num_threads(num_threads_);
    }
    const bool has_progress = !progress_callback.is_none() && progress_every > 0;

    for (int m = 0; m < n_merges; ++m) {
        const auto& [a, b, merged] = ms[m];
        {
            py::gil_scoped_release release;
            #pragma omp parallel for schedule(dynamic, 64)
            for (int cid = 0; cid < n_chunks; ++cid) {
                replay_single_merge_one_chunk(symbols_per_chunk_[cid], a, b, merged);
            }
        }
        if (has_progress && (m + 1) % progress_every == 0) {
            progress_callback(m + 1, n_merges);
        }
    }
}

std::unordered_map<uint64_t, int64_t> BpeEngine::apply_merge_internal(
    int32_t a, int32_t b, int32_t merged_id) {
    uint64_t pair_key = pack_pair(a, b);

    // Snapshot the chunks containing (a,b), then drop the entry —
    // it's about to be gone.
    std::vector<int32_t> affected;
    {
        auto it = where_.find(pair_key);
        if (it != where_.end()) {
            affected = std::move(it->second);
            where_.erase(it);
        }
    }
    pair_counts_.erase(pair_key);

    const int n = static_cast<int>(affected.size());
    if (num_threads_ > 0) {
        omp_set_num_threads(num_threads_);
    }

    const int actual_threads = num_threads_ > 0 ? num_threads_ : omp_get_max_threads();
    std::vector<LocalDeltas> per_thread(actual_threads);

    {
        py::gil_scoped_release release;
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            LocalDeltas& local = per_thread[tid];
            #pragma omp for schedule(dynamic, 64)
            for (int idx = 0; idx < n; ++idx) {
                int32_t cid = affected[idx];
                apply_merge_one_chunk(symbols_per_chunk_[cid],
                                      cid,
                                      chunk_weights_[cid],
                                      a, b, merged_id, local);
            }
        }
    }

    // Reduce thread-local deltas.
    std::unordered_map<uint64_t, int64_t> total_deltas;
    for (auto& local : per_thread) {
        for (auto& [key, dv] : local.count_deltas) {
            total_deltas[key] += dv;
        }
    }
    for (auto& local : per_thread) {
        for (auto& [key, cid] : local.where_adds) {
            where_[key].push_back(cid);
        }
    }

    // Fold per-pair deltas into pair_counts_; erase pairs whose new count
    // is <= 0. Caller decides what to do with the returned map (build a
    // py::list for the public path, or push to the native heap for the
    // run_merge_loop path).
    for (auto& [key, dv] : total_deltas) {
        int64_t new_count = pair_counts_[key] + dv;
        if (new_count <= 0) {
            pair_counts_.erase(key);
        } else {
            pair_counts_[key] = new_count;
        }
    }

    return total_deltas;
}

py::list BpeEngine::apply_merge(int32_t a, int32_t b, int32_t merged_id) {
    auto total_deltas = apply_merge_internal(a, b, merged_id);
    py::list out;
    for (auto& [key, dv] : total_deltas) {
        out.append(py::make_tuple(unpack_a(key), unpack_b(key), dv));
    }
    return out;
}

void BpeEngine::drop_pair(int32_t a, int32_t b) {
    uint64_t key = pack_pair(a, b);
    pair_counts_.erase(key);
    where_.erase(key);
}

std::string BpeEngine::id2sym(int32_t id) const {
    if (id < 0 || id >= static_cast<int32_t>(id2sym_native_.size())) {
        throw std::out_of_range("id out of range");
    }
    return id2sym_native_[id];
}

int32_t BpeEngine::sym2id(const std::string& sym) const {
    auto it = vocab_native_.find(sym);
    if (it == vocab_native_.end()) {
        throw std::out_of_range("symbol not in native vocab");
    }
    return it->second;
}

int BpeEngine::run_merge_loop(int32_t target_vocab_size,
                             py::object merge_filter,
                             py::object progress_callback,
                             int32_t progress_every) {
    // Seed the heap from current pair_counts_.
    heap_.clear();
    heap_.reserve(pair_counts_.size());
    for (const auto& [key, count] : pair_counts_) {
        heap_.push_back({-count, key});
    }
    std::make_heap(heap_.begin(), heap_.end(), std::greater<HeapEntry>{});

    const bool has_filter = !merge_filter.is_none();
    const bool has_progress = !progress_callback.is_none() && progress_every > 0;
    int n_accepted = 0;

    while (static_cast<int32_t>(id2sym_native_.size()) < target_vocab_size) {
        // Pop until live entry.
        HeapEntry top{};
        bool found_live = false;
        while (!heap_.empty()) {
            top = heap_.front();
            std::pop_heap(heap_.begin(), heap_.end(), std::greater<HeapEntry>{});
            heap_.pop_back();
            int64_t count = -top.neg_count;
            auto it = pair_counts_.find(top.pair_key);
            int64_t cur = (it == pair_counts_.end()) ? 0 : it->second;
            if (cur == count && cur > 0) { found_live = true; break; }
        }
        if (!found_live) break;

        int32_t a = unpack_a(top.pair_key);
        int32_t b = unpack_b(top.pair_key);
        std::string merged_s = id2sym_native_[a] + id2sym_native_[b];

        // Optional Python merge_filter callback. False → veto.
        if (has_filter) {
            bool accept = py::cast<bool>(
                merge_filter(id2sym_native_[a], id2sym_native_[b], merged_s));
            if (!accept) {
                uint64_t key = pack_pair(a, b);
                pair_counts_.erase(key);
                where_.erase(key);
                continue;
            }
        }

        // Accept the merge.
        int32_t merged_id = static_cast<int32_t>(id2sym_native_.size());
        id2sym_native_.push_back(merged_s);
        vocab_native_[merged_s] = merged_id;
        merges_native_.emplace_back(a, b);

        // Apply the merge to symbols_per_chunk_, fold deltas into
        // pair_counts_ + where_, push new heap entries for each changed pair.
        auto deltas = apply_merge_internal(a, b, merged_id);
        for (const auto& [key, _dv] : deltas) {
            auto it = pair_counts_.find(key);
            if (it == pair_counts_.end()) continue;  // pair is gone
            heap_.push_back({-it->second, key});
            std::push_heap(heap_.begin(), heap_.end(), std::greater<HeapEntry>{});
        }

        ++n_accepted;

        // Progress callback when total vocab size hits a multiple of
        // progress_every — matches Python's `len(vocab) % progress_every == 0`.
        if (has_progress
         && static_cast<int32_t>(id2sym_native_.size()) % progress_every == 0) {
            py::dict snapshot_vocab = get_vocab();
            py::list snapshot_merges = get_merges();
            progress_callback(static_cast<int>(id2sym_native_.size()),
                              snapshot_vocab, snapshot_merges);
        }
    }

    return n_accepted;
}

py::dict BpeEngine::get_vocab() const {
    py::dict out;
    for (size_t i = 0; i < id2sym_native_.size(); ++i) {
        out[py::str(id2sym_native_[i])] = static_cast<int32_t>(i);
    }
    return out;
}

py::list BpeEngine::get_merges() const {
    py::list out;
    for (const auto& [a, b] : merges_native_) {
        out.append(py::make_tuple(id2sym_native_[a], id2sym_native_[b]));
    }
    return out;
}
