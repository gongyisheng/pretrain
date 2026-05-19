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

// Single-merge left-to-right collapse for one chunk.
inline void replay_single_merge_one_chunk(
    std::vector<int32_t>& chunk,
    int32_t a,
    int32_t b,
    int32_t merged
) {
    if (chunk.size() < 2) return;
    size_t i = 0;
    while (i + 1 < chunk.size()) {
        if (chunk[i] == a && chunk[i + 1] == b) {
            chunk[i] = merged;
            chunk.erase(chunk.begin() + static_cast<long>(i) + 1);
            ++i;
        } else {
            ++i;
        }
    }
}

struct LocalDeltas {
    std::vector<std::pair<uint64_t, int64_t>> count_deltas;  // (packed pair, delta_value)
    std::vector<std::pair<uint64_t, int32_t>> chunks_by_pair_adds;    // (packed pair, chunk_id)
};

// Apply caller-supplied thread count to OMP and return the effective count.
// num_threads <= 0 leaves OMP at its default; the return is sized for
// per-thread buffer allocation.
int configure_omp_threads(int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
        return num_threads;
    }
    return omp_get_max_threads();
}

inline bool apply_merge_one_chunk(
    std::vector<int32_t>& chunk,
    int32_t chunk_id,
    int64_t chunk_count,
    int32_t a,
    int32_t b,
    int32_t merged,
    LocalDeltas& out
) {
    if (chunk.size() < 2) return false;
    // Dedupe pair keys per chunk
    std::unordered_set<uint64_t> seen_in_chunk;
    bool changed = false;
    size_t i = 0;
    while (i + 1 < chunk.size()) {
        if (chunk[i] != a || chunk[i + 1] != b) {
            ++i;
            continue;
        }
        changed = true;

        // deal with (prev, a)
        if (i > 0) {
            int32_t prev = chunk[i - 1];
            out.count_deltas.emplace_back(pack_pair(prev, a), -chunk_count);
            uint64_t k_pm = pack_pair(prev, merged);
            out.count_deltas.emplace_back(k_pm, chunk_count);
            if (seen_in_chunk.insert(k_pm).second) {
                out.chunks_by_pair_adds.emplace_back(k_pm, chunk_id);
            }
        }

        // deal with (b, next)
        if (i + 2 < chunk.size()) {
            int32_t nxt = chunk[i + 2];
            out.count_deltas.emplace_back(pack_pair(b, nxt), -chunk_count);
            uint64_t k_mn = pack_pair(merged, nxt);
            out.count_deltas.emplace_back(k_mn, chunk_count);
            if (seen_in_chunk.insert(k_mn).second) {
                out.chunks_by_pair_adds.emplace_back(k_mn, chunk_id);
            }
        }

        // deal with (a, b)
        chunk[i] = merged;
        chunk.erase(chunk.begin() + static_cast<long>(i) + 1);
        ++i;
    }
    return changed;
}

}  // namespace

void BpeEngine::load_chunks(py::dict chunks, py::dict vocab) {

    std::vector<std::pair<std::vector<int32_t>, int64_t>> resolved;
    resolved.reserve(chunks.size());

    for (auto item : chunks) {
        py::tuple tup = py::reinterpret_borrow<py::tuple>(item.first);
        int64_t chunk_count = py::cast<int64_t>(item.second);

        std::vector<int32_t> ids;
        ids.reserve(tup.size());
        for (auto token : tup) {
            // Explicit check so the KeyError names the offending token AND
            // the chunk that referenced it; the implicit vocab[token] miss
            // would surface only the bare token repr.
            if (!vocab.contains(token)) {
                std::string token_repr = py::cast<std::string>(py::repr(token));
                std::string chunk_repr = py::cast<std::string>(py::repr(tup));
                throw py::key_error(
                    "token " + token_repr + " not in vocab "
                    "(referenced by chunk " + chunk_repr + ")"
                );
            }
            py::object id_obj = vocab[token];
            ids.push_back(py::cast<int32_t>(id_obj));
        }
        resolved.emplace_back(std::move(ids), chunk_count);
    }

    std::sort(resolved.begin(), resolved.end(),
    [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });

    tokens_by_chunk_.clear();
    chunk_counts_.clear();
    tokens_by_chunk_.reserve(resolved.size());
    chunk_counts_.reserve(resolved.size());
    for (auto& [ids, chunk_count] : resolved) {
        tokens_by_chunk_.push_back(std::move(ids));
        chunk_counts_.push_back(chunk_count);
    }

    id2token_.clear();
    token2id_.clear();
    int32_t max_id = -1;
    for (auto item : vocab) {
        int32_t id = py::cast<int32_t>(item.second);
        if (id > max_id) max_id = id;
    }
    id2token_.assign(max_id + 1, std::string{});
    for (auto item : vocab) {
        std::string token = py::cast<std::string>(item.first);
        int32_t id = py::cast<int32_t>(item.second);
        id2token_[id] = token;
        token2id_[token] = id;
    }
}

std::vector<int32_t> BpeEngine::get_chunk_tokens(int32_t chunk_id) const {
    if (chunk_id < 0 || chunk_id >= static_cast<int32_t>(tokens_by_chunk_.size())) {
        throw std::out_of_range("chunk_id out of range");
    }
    return tokens_by_chunk_[chunk_id];
}

int64_t BpeEngine::get_chunk_count(int32_t chunk_id) const {
    if (chunk_id < 0 || chunk_id >= static_cast<int32_t>(chunk_counts_.size())) {
        throw std::out_of_range("chunk_id out of range");
    }
    return chunk_counts_[chunk_id];
}

py::list BpeEngine::build_pair_index() {
    pair_counts_.clear();
    chunks_by_pair_.clear();

    const int32_t n = static_cast<int32_t>(tokens_by_chunk_.size());
    for (int32_t cid = 0; cid < n; ++cid) {
        const auto& chunk = tokens_by_chunk_[cid];
        const int64_t chunk_count = chunk_counts_[cid];
        if (chunk.size() < 2) continue;
        // Dedupe so chunks_by_pair_[pair] doesn't list the same chunk twice.
        std::unordered_set<uint64_t> seen_in_chunk;
        for (size_t i = 0; i + 1 < chunk.size(); ++i) {
            uint64_t key = pack_pair(chunk[i], chunk[i + 1]);
            pair_counts_[key] += chunk_count;
            if (seen_in_chunk.insert(key).second) {
                chunks_by_pair_[key].push_back(cid);
            }
        }
    }

    py::list out;
    for (const auto& [key, count] : pair_counts_) {
        out.append(py::make_tuple(unpack_a(key), unpack_b(key), count));
    }
    return out;
}

int64_t BpeEngine::get_pair_count(int32_t a, int32_t b) const {
    auto it = pair_counts_.find(pack_pair(a, b));
    if (it == pair_counts_.end()) return 0;
    return it->second;
}

std::vector<int32_t> BpeEngine::get_chunks_by_pair(int32_t a, int32_t b) const {
    auto it = chunks_by_pair_.find(pack_pair(a, b));
    if (it == chunks_by_pair_.end()) return {};
    return it->second;
}

void BpeEngine::set_num_threads(int n) {
    num_threads_ = n;
}

void BpeEngine::run_replay_merges(
    py::list merges,
    py::object progress_callback,
    int32_t progress_every
) {

    std::vector<std::tuple<int32_t, int32_t, int32_t>> ms;
    ms.reserve(merges.size());
    merges_.reserve(merges_.size() + merges.size());
    for (auto item : merges) {
        py::tuple t = py::reinterpret_borrow<py::tuple>(item);
        int32_t a = py::cast<int32_t>(t[0]);
        int32_t b = py::cast<int32_t>(t[1]);
        int32_t merged = py::cast<int32_t>(t[2]);
        ms.emplace_back(a, b, merged);
        merges_.emplace_back(a, b);
    }

    const int n_chunks = static_cast<int>(tokens_by_chunk_.size());
    const int n_merges = static_cast<int>(ms.size());
    configure_omp_threads(num_threads_);
    const bool has_progress = !progress_callback.is_none() && progress_every > 0;

    // keeps the token vector hot in L1/L2 across the batch
    const int batch_size = progress_every > 0 ? progress_every : n_merges;
    for (int m_start = 0; m_start < n_merges; m_start += batch_size) {
        if (PyErr_CheckSignals() != 0) throw py::error_already_set();
        const int m_end = std::min(m_start + batch_size, n_merges);
        {
            py::gil_scoped_release release;
            #pragma omp parallel for schedule(dynamic, 64)
            for (int cid = 0; cid < n_chunks; ++cid) {
                for (int m = m_start; m < m_end; ++m) {
                    const auto& [a, b, merged] = ms[m];
                    replay_single_merge_one_chunk(tokens_by_chunk_[cid], a, b, merged);
                }
            }
        }
        if (has_progress) {
            progress_callback(m_end, n_merges);
        }
    }
}

std::unordered_map<uint64_t, int64_t> BpeEngine::apply_merge_internal(
    int32_t a, 
    int32_t b, 
    int32_t merged_id
) {
    uint64_t pair_key = pack_pair(a, b);

    // Snapshot affected chunks, then drop the (a,b) entry.
    std::vector<int32_t> affected;
    {
        auto it = chunks_by_pair_.find(pair_key);
        if (it != chunks_by_pair_.end()) {
            affected = std::move(it->second);
            chunks_by_pair_.erase(it);
        }
    }
    pair_counts_.erase(pair_key);

    const int n = static_cast<int>(affected.size());
    const int actual_threads = configure_omp_threads(num_threads_);
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
                apply_merge_one_chunk(
                    tokens_by_chunk_[cid],
                    cid,
                    chunk_counts_[cid],
                    a, b, merged_id, local);
            }
        }
    }

    std::unordered_map<uint64_t, int64_t> total_deltas;
    for (auto& local : per_thread) {
        for (auto& [key, dv] : local.count_deltas) {
            total_deltas[key] += dv;
        }
    }
    for (auto& local : per_thread) {
        for (auto& [key, cid] : local.chunks_by_pair_adds) {
            chunks_by_pair_[key].push_back(cid);
        }
    }

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
    chunks_by_pair_.erase(key);
}

std::string BpeEngine::id2token(int32_t id) const {
    if (id < 0 || id >= static_cast<int32_t>(id2token_.size())) {
        throw std::out_of_range("id out of range");
    }
    return id2token_[id];
}

int32_t BpeEngine::token2id(const std::string& token) const {
    auto it = token2id_.find(token);
    if (it == token2id_.end()) {
        throw std::out_of_range("token not in native vocab");
    }
    return it->second;
}

int BpeEngine::run_merge_loop(
    int32_t target_vocab_size,
    py::object merge_filter,
    py::object progress_callback,
    int32_t progress_every
) {
    heap_.clear();
    heap_.reserve(pair_counts_.size());
    for (const auto& [key, count] : pair_counts_) {
        heap_.push_back({-count, key});
    }
    std::make_heap(heap_.begin(), heap_.end(), std::greater<HeapEntry>{});

    const bool has_filter = !merge_filter.is_none();
    const bool has_progress = !progress_callback.is_none() && progress_every > 0;
    int n_accepted = 0;

    while (static_cast<int32_t>(id2token_.size()) < target_vocab_size) {
        // Ctrl+C check (GIL held).
        if (PyErr_CheckSignals() != 0) throw py::error_already_set();
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
        std::string merged_s = id2token_[a] + id2token_[b];

        // merge_filter returning False vetoes the merge.
        if (has_filter) {
            bool accept = py::cast<bool>(
                merge_filter(id2token_[a], id2token_[b], merged_s));
            if (!accept) {
                uint64_t key = pack_pair(a, b);
                pair_counts_.erase(key);
                chunks_by_pair_.erase(key);
                continue;
            }
        }

        int32_t merged_id = static_cast<int32_t>(id2token_.size());
        id2token_.push_back(merged_s);
        token2id_[merged_s] = merged_id;
        merges_.emplace_back(a, b);

        // Push heap entries for each pair whose count changed.
        auto deltas = apply_merge_internal(a, b, merged_id);
        for (const auto& [key, _dv] : deltas) {
            auto it = pair_counts_.find(key);
            if (it == pair_counts_.end()) continue;  // pair is gone
            heap_.push_back({-it->second, key});
            std::push_heap(heap_.begin(), heap_.end(), std::greater<HeapEntry>{});
        }

        ++n_accepted;

        if (has_progress
         && static_cast<int32_t>(id2token_.size()) % progress_every == 0) {
            py::dict snapshot_vocab = get_vocab();
            py::list snapshot_merges = get_merges();
            progress_callback(static_cast<int>(id2token_.size()),
                              snapshot_vocab, snapshot_merges);
        }
    }

    return n_accepted;
}

py::dict BpeEngine::get_vocab() const {
    py::dict out;
    for (size_t i = 0; i < id2token_.size(); ++i) {
        out[py::str(id2token_[i])] = static_cast<int32_t>(i);
    }
    return out;
}

py::list BpeEngine::get_merges() const {
    py::list out;
    for (const auto& [a, b] : merges_) {
        out.append(py::make_tuple(id2token_[a], id2token_[b]));
    }
    return out;
}
