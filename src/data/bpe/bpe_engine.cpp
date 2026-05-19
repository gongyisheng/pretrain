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

int set_omp_threads(int num_threads) {
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);
    return num_threads;
}

}  // namespace

uint64_t Pair::pack(int32_t a, int32_t b) {
    return (
        static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
        static_cast<uint64_t>(static_cast<uint32_t>(b)
    );
}

std::pair<int32_t, int32_t> Pair::unpack(uint64_t key) {
    return {
        static_cast<int32_t>(static_cast<uint32_t>(key >> 32)),
        static_cast<int32_t>(static_cast<uint32_t>(key & 0xFFFFFFFFu))
    };
}

void ThreadLocalDelta::add_pair_delta(uint64_t pair_key, int64_t delta) {
    pair_count_deltas.emplace_back(pair_key, delta);
}

void ThreadLocalDelta::add_chunk_delta(uint64_t pair_key, int32_t chunk_id) {
    chunk_id_adds.emplace_back(pair_key, chunk_id);
}

bool BpeEngine::HeapEntry::operator>(const HeapEntry& other) const {
    if (neg_count != other.neg_count) return neg_count > other.neg_count;
    return pair_key > other.pair_key;
}

int32_t BpeEngine::get_num_chunks() const {
    return static_cast<int32_t>(chunks_.size());
}

int32_t BpeEngine::get_vocab_size() const {
    return static_cast<int32_t>(id2token_.size());
}

bool BpeEngine::merge_one_chunk_(
    Chunk& chunk,
    int32_t a,
    int32_t b,
    int32_t merged,
    ThreadLocalDelta* out
) {
    auto& tokens = chunk.tokens;
    if (tokens.size() < 2) return false;
    bool changed = false;
    // Dedupe new-pair chunk_id emissions within this chunk so Pair::chunk_ids
    // (a vector, not a set) does not accumulate duplicates.
    std::unordered_set<uint64_t> seen_new_pairs;
    size_t i = 0;
    while (i + 1 < tokens.size()) {
        if (tokens[i] != a || tokens[i + 1] != b) {
            ++i;
            continue;
        }
        changed = true;

        if (out) {
            const int32_t chunk_id = chunk.id;
            const int64_t chunk_count = chunk.count;

            // deal with (prev, a) -> (prev, merged)
            if (i > 0) {
                int32_t prev = tokens[i - 1];
                out->add_pair_delta(Pair::pack(prev, a), -chunk_count);
                uint64_t k_pm = Pair::pack(prev, merged);
                out->add_pair_delta(k_pm, chunk_count);
                if (seen_new_pairs.insert(k_pm).second) {
                    out->add_chunk_delta(k_pm, chunk_id);
                }
            }

            // deal with (b, next) -> (merged, next)
            if (i + 2 < tokens.size()) {
                int32_t nxt = tokens[i + 2];
                out->add_pair_delta(Pair::pack(b, nxt), -chunk_count);
                uint64_t k_mn = Pair::pack(merged, nxt);
                out->add_pair_delta(k_mn, chunk_count);
                if (seen_new_pairs.insert(k_mn).second) {
                    out->add_chunk_delta(k_mn, chunk_id);
                }
            }
        }

        // deal with (a, b)
        tokens[i] = merged;
        tokens.erase(tokens.begin() + static_cast<long>(i) + 1);
        ++i;
    }
    return changed;
}

// Load pretokenized corpus + vocab into the engine.
//   chunks: Dict[Tuple[str, ...], int]
//       key   = one pretokenized word's token sequence.
//       value = corpus occurrence count of that sequence.
//   vocab:  Dict[str, int]
//       key   = token string (must contain every token referenced by chunks).
//       value = int32 token id.
//
// Example (corpus "hi hi hi hit hit", char-level vocab):
//   chunks = {("h", "i"): 3, ("h", "i", "t"): 2}
//   vocab  = {"h": 0, "i": 1, "t": 2}
void BpeEngine::init_chunks_(py::dict chunks, py::dict vocab) {

    std::vector<std::pair<std::vector<int32_t>, int64_t>> resolved;
    resolved.reserve(chunks.size());

    for (auto item : chunks) {
        py::tuple chunk_tuple = py::reinterpret_borrow<py::tuple>(item.first);
        int64_t chunk_count = py::cast<int64_t>(item.second);

        std::vector<int32_t> ids;
        ids.reserve(chunk_tuple.size());
        for (auto token : chunk_tuple) {
            // Raise key_error if token not in
            if (!vocab.contains(token)) {
                std::string token_repr = py::cast<std::string>(py::repr(token));
                std::string chunk_repr = py::cast<std::string>(py::repr(chunk_tuple));
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

    // sorted lexicographically to ensure reproducibility
    std::sort(resolved.begin(), resolved.end(),
    [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });

    // init chunks
    chunks_.clear();
    chunks_.reserve(resolved.size());
    for (size_t i = 0; i < resolved.size(); ++i) {
        auto& [ids, chunk_count] = resolved[i];
        chunks_.push_back(Chunk{static_cast<int32_t>(i), std::move(ids), chunk_count});
    }

    // init id <-> token map
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
    if (chunk_id < 0 || chunk_id >= static_cast<int32_t>(chunks_.size())) {
        throw std::out_of_range("chunk_id out of range");
    }
    return chunks_[chunk_id].tokens;
}

int64_t BpeEngine::get_chunk_count(int32_t chunk_id) const {
    if (chunk_id < 0 || chunk_id >= static_cast<int32_t>(chunks_.size())) {
        throw std::out_of_range("chunk_id out of range");
    }
    return chunks_[chunk_id].count;
}

void BpeEngine::init_pairs_() {
    pairs_.clear();

    std::unordered_set<uint64_t> seen_in_chunk;

    const int32_t n = static_cast<int32_t>(chunks_.size());
    for (int32_t cid = 0; cid < n; ++cid) {
        const auto& chunk = chunks_[cid];
        if (chunk.tokens.size() < 2) continue;
        seen_in_chunk.clear();
        for (size_t i = 0; i + 1 < chunk.tokens.size(); ++i) {
            int32_t a = chunk.tokens[i];
            int32_t b = chunk.tokens[i + 1];
            uint64_t key = Pair::pack(a, b);
            auto [it, inserted] = pairs_.try_emplace(key);
            if (inserted) {
                it->second.a = a;
                it->second.b = b;
                it->second.count = 0;
            }
            it->second.count += chunk.count;
            if (seen_in_chunk.insert(key).second) {
                it->second.chunk_ids.push_back(cid);
            }
        }
    }
}

py::list BpeEngine::list_pairs() const {
    py::list out;
    for (const auto& [key, p] : pairs_) {
        out.append(py::make_tuple(p.a, p.b, p.count));
    }
    return out;
}

// Public entry point: load corpus, build initial pair index.
void BpeEngine::feed(py::dict chunks, py::dict vocab) {
    init_chunks_(chunks, vocab);
    init_pairs_();
}

// Public entry point: replay pre-computed merges, then refresh pair index
// so the engine is ready for train().
void BpeEngine::replay_merges(
    py::list merges,
    py::object progress_callback,
    int32_t progress_every
) {
    replay_merges_(merges, progress_callback, progress_every);
    init_pairs_();
}

int64_t BpeEngine::get_pair_count(int32_t a, int32_t b) const {
    auto it = pairs_.find(Pair::pack(a, b));
    if (it == pairs_.end()) return 0;
    return it->second.count;
}

std::vector<int32_t> BpeEngine::get_chunks_by_pair(int32_t a, int32_t b) const {
    auto it = pairs_.find(Pair::pack(a, b));
    if (it == pairs_.end()) return {};
    return it->second.chunk_ids;
}

void BpeEngine::set_num_threads(int n) {
    num_threads_ = n;
}

// Apply a pre-computed merge list to every chunk (tokens mutated in place).
//   merges: List[Tuple[int, int, int]]
//       Each tuple is (a_id, b_id, merged_id):
//         a_id, b_id   = IDs of the two adjacent tokens to collapse.
//         merged_id    = pre-assigned ID for the merged token (caller knows
//                        this from the saved tokenizer's vocab).
//
// Example (replay "h"+"i" → "hi" where "hi" already has id 256):
//   merges = [(104, 105, 256)]
//   chunk before: [104, 105]    chunk after: [256]
void BpeEngine::replay_merges_(
    py::list merges,
    py::object progress_callback,
    int32_t progress_every
) {

    // copy for releasing GIL
    std::vector<std::tuple<int32_t, int32_t, int32_t>> merges_copy;
    merges_copy.reserve(merges.size());
    merges_.reserve(merges_.size() + merges.size());
    for (auto item : merges) {
        py::tuple t = py::reinterpret_borrow<py::tuple>(item);
        int32_t a = py::cast<int32_t>(t[0]);
        int32_t b = py::cast<int32_t>(t[1]);
        int32_t merged = py::cast<int32_t>(t[2]);
        merges_copy.emplace_back(a, b, merged);
        merges_.emplace_back(a, b);
    }

    const int n_chunks = static_cast<int>(chunks_.size());
    const int n_merges = static_cast<int>(merges_copy.size());

    set_omp_threads(num_threads_);
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
                    const auto& [a, b, merged] = merges_copy[m];
                    merge_one_chunk_(chunks_[cid], a, b, merged);
                }
            }
        }
        if (has_progress) {
            progress_callback(m_end, n_merges);
        }
    }
}

std::unordered_map<uint64_t, int64_t> BpeEngine::apply_merge_(
    int32_t a,
    int32_t b,
    int32_t merged_id
) {
    uint64_t pair_key = Pair::pack(a, b);

    // Snapshot affected chunks, then drop the (a,b) entry. Move the vector
    // out before erase to avoid an extra copy.
    std::vector<int32_t> affected;
    {
        auto it = pairs_.find(pair_key);
        if (it != pairs_.end()) {
            affected = std::move(it->second.chunk_ids);
            pairs_.erase(it);
        }
    }

    const int n = static_cast<int>(affected.size());
    const int actual_threads = set_omp_threads(num_threads_);
    std::vector<ThreadLocalDelta> per_thread(actual_threads);

    {
        py::gil_scoped_release release;
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            ThreadLocalDelta& local = per_thread[tid];
            #pragma omp for schedule(dynamic, 64)
            for (int idx = 0; idx < n; ++idx) {
                int32_t cid = affected[idx];
                merge_one_chunk_(chunks_[cid], a, b, merged_id, &local);
            }
        }
    }

    // Sum-reduce pair count deltas across threads.
    std::unordered_map<uint64_t, int64_t> total_pair_count_deltas;
    for (auto& local : per_thread) {
        for (auto& [key, dv] : local.pair_count_deltas) {
            total_pair_count_deltas[key] += dv;
        }
    }

    // apply count deltas
    for (auto& [key, val] : total_pair_count_deltas) {
        auto it = pairs_.find(key);
        if (it == pairs_.end()) {
            // create new pair if pair doesn't exist
            if (val > 0) {
                Pair p;
                auto [pa, pb] = Pair::unpack(key);
                p.a = pa;
                p.b = pb;
                p.count = val;
                pairs_.emplace(key, std::move(p));
            }
        } else {
            // if pair exist, change delta value or erase
            it->second.count += val;
            if (it->second.count <= 0) {
                pairs_.erase(it);
            }
        }
    }

    // apply chunk_id deltas
    for (auto& local : per_thread) {
        for (auto& [key, cid] : local.chunk_id_adds) {
            auto it = pairs_.find(key);
            if (it != pairs_.end()) {
                it->second.chunk_ids.push_back(cid);
            }
        }
    }

    return total_pair_count_deltas;
}

py::list BpeEngine::apply_merge(int32_t a, int32_t b, int32_t merged_id) {
    auto total_deltas = apply_merge_(a, b, merged_id);
    py::list out;
    for (auto& [key, dv] : total_deltas) {
        auto [a, b] = Pair::unpack(key);
        out.append(py::make_tuple(a, b, dv));
    }
    return out;
}

void BpeEngine::drop_pair(int32_t a, int32_t b) {
    pairs_.erase(Pair::pack(a, b));
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

int BpeEngine::train(
    int32_t target_vocab_size,
    py::object merge_filter,
    py::object progress_callback,
    int32_t progress_every
) {
    heap_.clear();
    heap_.reserve(pairs_.size());
    for (const auto& [key, p] : pairs_) {
        heap_.push_back({-p.count, key});
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
            auto it = pairs_.find(top.pair_key);
            int64_t cur = (it == pairs_.end()) ? 0 : it->second.count;
            if (cur == count && cur > 0) { found_live = true; break; }
        }
        if (!found_live) break;

        auto [a, b] = Pair::unpack(top.pair_key);
        std::string merged_token = id2token_[a] + id2token_[b];

        // merge_filter returning False vetoes the merge.
        if (has_filter) {
            bool accept = py::cast<bool>(
                merge_filter(id2token_[a], id2token_[b], merged_token));
            if (!accept) {
                pairs_.erase(top.pair_key);
                continue;
            }
        }

        int32_t merged_id = static_cast<int32_t>(id2token_.size());
        id2token_.push_back(merged_token);
        token2id_[merged_token] = merged_id;
        merges_.emplace_back(a, b);

        // Push heap entries for each pair whose count changed.
        auto deltas = apply_merge_(a, b, merged_id);
        for (const auto& [key, _dv] : deltas) {
            auto it = pairs_.find(key);
            if (it == pairs_.end()) continue;  // pair is gone
            heap_.push_back({-it->second.count, key});
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
