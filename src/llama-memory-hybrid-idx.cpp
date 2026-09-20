#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <atomic>
#include <cstdio>
#include <cstdlib>

// [TAG_QSA_DBG] ---------------------------------------------------------------------------
// Temporary instrumentation for the pooled-cache sizing/fill mismatch:
//   qsa_pooled_n_dirty_max() (graph build) sizes the dirty tables from the ubatch's
//   temporal positions, while set_input_qsa() (decode) derives n_complete from the
//   sequence's actual cells. When the fill needs more rows than the graph allocated,
//   GGML_ASSERT aborts. The graph-build estimate is stashed here so the failing
//   ubatch can log both sides. Remove this block with the rest of [TAG_QSA_DBG].
namespace {
struct qsa_dbg_est {
    std::atomic<int64_t> n_dirty_max{-1};
    std::atomic<int64_t> n_complete{-1};
    std::atomic<int64_t> w{-1};
    std::atomic<int64_t> q_max{-1};
    std::atomic<int64_t> n_tokens{-1};
    std::atomic<int64_t> ratio{-1};
    std::atomic<int64_t> seq{-1};
    std::atomic<int64_t> calls{0};
    std::atomic<int64_t> mock{0};
};

qsa_dbg_est          g_qsa_dbg_est;
std::atomic<int64_t> g_qsa_dbg_hits{0};
std::atomic<int64_t> g_qsa_dbg_spam{0};

void qsa_dbg_note(uint32_t n_dirty_max, int64_t n_complete, int64_t w, int64_t q_max,
                  int64_t n_tokens, uint32_t ratio, int64_t seq, bool is_mock) {
    g_qsa_dbg_est.n_dirty_max.store((int64_t) n_dirty_max);
    g_qsa_dbg_est.n_complete .store(n_complete);
    g_qsa_dbg_est.w          .store(w);
    g_qsa_dbg_est.q_max      .store(q_max);
    g_qsa_dbg_est.n_tokens   .store(n_tokens);
    g_qsa_dbg_est.ratio      .store((int64_t) ratio);
    g_qsa_dbg_est.seq        .store(seq);
    g_qsa_dbg_est.mock       .store(is_mock ? 1 : 0);
    g_qsa_dbg_est.calls.fetch_add(1);
}
} // namespace
// ---------------------------------------------------------------------------------------


//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    // [TAG_QSA_POOLED_CACHE] one f32 row per position block per layer; single-stream memories
    // only (a unified cache shares one stream; block rows are position-indexed)
    if (mem_idx && mem_idx->get_n_stream() == 1) {
        uint32_t ratio = 0;
        for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
            // only the dense-attention layers own a QSA cache: a recurrent layer may still
            // carry a nonzero ratio in the metadata, but it is not a pooled-cache layer
            if (!model.hparams.is_recr(il) && model.hparams.dsv4_compress_ratios[il] > 0) {
                ratio = model.hparams.dsv4_compress_ratios[il];
                break;
            }
        }

        const uint32_t idx_dim = model.hparams.indexer_head_size;

        if (ratio > 0 && idx_dim > 0) {
            // + 1 so a partial trailing block has a slot, + 1 dustbin row for padded writes
            // [TAG_QSA_OWNROW] rows are laid out per sequence: sequences sharing one unified
            // stream hold different cells at the same positions, so one row per position block
            // cannot serve them all - each would score the other summaries. pooled_rows stays
            // the total (allocation + views); a sequence's rows start at pooled_row_base().
            pooled_rows_per_seq = kv_size/ratio + 2;
            pooled_n_seq_max    = std::max<uint32_t>(1, n_seq_max);
            pooled_rows         = pooled_rows_per_seq * pooled_n_seq_max;
            pooled_ratio        = ratio;

            // one context+buffer per device: the indexer caches of the QSA layers are spread
            // across the layer-split devices, and a row written by a device that does not own
            // it would travel the inter-GPU link every decode step
            std::vector<ggml_backend_buffer_type_t>   bufts;
            std::vector<std::vector<ggml_tensor *>>   per_buf_tensors;

            for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
                // the idx cache is filtered to the dense-attention layers; get_k_storage on a
                // recurrent layer is out of range even when it carries a ratio
                if (model.hparams.is_recr(il) || model.hparams.dsv4_compress_ratios[il] == 0) {
                    continue;
                }
                ggml_tensor * k = mem_idx->get_k_storage((int32_t) il);
                if (k == nullptr) {
                    continue;
                }

                const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(k->buffer);

                size_t ci = SIZE_MAX;
                for (size_t j = 0; j < bufts.size(); ++j) {
                    if (bufts[j] == buft) {
                        ci = j;
                        break;
                    }
                }
                if (ci == SIZE_MAX) {
                    ci = bufts.size();
                    bufts.push_back(buft);
                    per_buf_tensors.emplace_back();

                    ggml_init_params ip = {
                        /*.mem_size   =*/ 2*model.hparams.n_layer()*ggml_tensor_overhead(),
                        /*.mem_buffer =*/ nullptr,
                        /*.no_alloc   =*/ true,
                    };
                    pooled_ctxs.emplace_back(ggml_init(ip));
                }

                ggml_tensor * t = ggml_new_tensor_2d(pooled_ctxs[ci].get(), GGML_TYPE_F32, idx_dim, pooled_rows);
                ggml_format_name(t, "idx_pooled_l%u", il);
                pooled_k[(int32_t) il] = t;
                per_buf_tensors[ci].push_back(t);
            }

            size_t total_bytes = 0;
            for (size_t ci = 0; ci < bufts.size(); ++ci) {
                pooled_bufs.emplace_back(ggml_backend_alloc_ctx_tensors_from_buft(pooled_ctxs[ci].get(), bufts[ci]));
                GGML_ASSERT(pooled_bufs.back() && "failed to allocate the pooled indexer key cache");
                // stale rows are read (and masked); they must be finite, never uninitialized
                ggml_backend_buffer_clear(pooled_bufs.back().get(), 0);
                total_bytes += ggml_backend_buffer_get_size(pooled_bufs.back().get());
            }

            if (!pooled_k.empty()) {
                LLAMA_LOG_INFO("%s: pooled indexer key cache, %zu layers x %u rows on %zu buffers, %.2f MiB\n",
                        __func__, pooled_k.size(), pooled_rows, pooled_bufs.size(),
                        total_bytes/1024.0/1024.0);
            }
        }
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }

    // [TAG_QSA_POOLED_CACHE]
    pooled_reset(-1);
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    // [TAG_QSA_POOLED_CACHE]
    pooled_rm(seq_id, p0, p1);

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }

    // [TAG_QSA_POOLED_CACHE] rows are shared in the single-stream cache; the copy's blocks
    // are refilled from its own cells on its first ubatch
    pooled_reset(seq_id_dst);
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }

    // [TAG_QSA_POOLED_CACHE] only seq_id's rows survive as trusted
    const int64_t keep = pooled_w.count(seq_id) ? pooled_w[seq_id] : 0;
    pooled_w.clear();
    pooled_w[seq_id] = keep;
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }

    // [TAG_QSA_POOLED_CACHE] shifting positions remaps every block
    pooled_reset(seq_id);
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }

    // [TAG_QSA_POOLED_CACHE]
    pooled_reset(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }

    // [TAG_QSA_POOLED_CACHE] a full restore rewrites the indexer cells with arbitrary
    // content, so no pooled row can be trusted; the next ubatch refills the whole range.
    // A PARTIAL_ONLY restore (speculative checkpoint replay) leaves the cells untouched
    // and its rollback arrives through seq_rm, which already clamped the watermark.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        pooled_reset(seq_id);
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }

    // [TAG_QSA_POOLED_CACHE]
    pooled_reset(seq_id);
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

ggml_tensor * llama_memory_hybrid_idx::get_pooled_k(int32_t il) const {
    const auto it = pooled_k.find(il);
    return it == pooled_k.end() ? nullptr : it->second;
}

int64_t & llama_memory_hybrid_idx::pooled_valid(llama_seq_id seq_id) const {
    return pooled_w[seq_id];
}

void llama_memory_hybrid_idx::pooled_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (pooled_k.empty()) {
        return;
    }

    if (p0 <= 0 && p1 < 0) {
        pooled_w[seq_id] = 0;
        return;
    }

    // blocks at or beyond the first removed position lose members; earlier rows keep their
    // content (removals only ever drop the tail or a middle range, never rewrite the prefix)
    const int64_t blk = pooled_ratio > 0 ? std::max<llama_pos>(p0, 0)/pooled_ratio : 0;

    auto & w = pooled_w[seq_id];
    w = std::min(w, blk);
}

void llama_memory_hybrid_idx::pooled_reset(llama_seq_id seq_id) {
    if (seq_id < 0) {
        pooled_w.clear();
    } else {
        pooled_w[seq_id] = 0;
    }
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        ggml_tensor * dirty_cells,
        ggml_tensor * dirty_pos,
        ggml_tensor * dirty_rows) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;
    // same formula as the graph; blk_pos may be null on the pooled path
    const int64_t n_blocks = (n_kv + r - 1)/r;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    // [TAG_QSA_BIAS] the block-level bias path indexes the bid arrays by block number; report the
    // cases where that correspondence does not hold (see the checks below). On by default,
    // LLAMA_QSA_BIAS=0 disables, capped at 12 lines per process.
    const char * bias_env = getenv("LLAMA_QSA_BIAS");
    const bool   bias_on  = !(bias_env != nullptr &&
            (bias_env[0] == '0' || bias_env[0] == 'o' || bias_env[0] == 'O' ||
             bias_env[0] == 'n' || bias_env[0] == 'N'));

    // [TAG_QSA_WINDOW] keep the last N position blocks always selectable, instead of only the
    // unpooled tail block. A sparse indexer that picks top_k blocks can otherwise leave the most
    // recent tokens (the user's instruction) out of the selected set entirely - the model then
    // answers as if no request was made. 0 restores the previous behaviour. The value is read from
    // ./qsa-window.txt on every call (so an A/B needs no restart) or from LLAMA_QSA_WINDOW when
    // set; it is in blocks, 1 block = ratio tokens.
    int64_t qsa_window_blocks = 0;
    {
        const char * wenv = getenv("LLAMA_QSA_WINDOW");

        if (wenv != nullptr) {
            qsa_window_blocks = atoll(wenv);
        } else if (FILE * wf = fopen("qsa-window.txt", "r")) {
            long long v = 0;

            if (fscanf(wf, "%lld", &v) == 1) {
                qsa_window_blocks = (int64_t) v;
            }

            fclose(wf);
        }

        if (qsa_window_blocks < 0) {
            qsa_window_blocks = 0;
        }

        if (qsa_window_blocks != pooled_window_logged) {
            pooled_window_logged = qsa_window_blocks;

            LLAMA_LOG_ERROR("[TAG_QSA_WINDOW] active: blocks=%lld (~%lld tokens of always-visible tail)\n",
                    (long long) qsa_window_blocks, (long long) (qsa_window_blocks * (int64_t) ratio));
        }
    }

    const int64_t qsa_window_tokens = qsa_window_blocks * (int64_t) ratio;

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    float   * dst_bias      = (float   *) bias->data;

    // [TAG_QSA_POOLED_CACHE] the pooled path drops blk_cells/blk_pos from the graph (the dirty
    // tables replace them), so they may be null here; the block map is still needed for the
    // dirty fill, so it is built in local buffers either way
    int32_t * dst_blk_cells = blk_cells != nullptr ? (int32_t *) blk_cells->data : nullptr;
    int32_t * dst_blk_pos   = blk_pos   != nullptr ? (int32_t *) blk_pos->data   : nullptr;

    std::vector<int32_t> loc_blk_cells(r*n_blocks);
    std::vector<int32_t> loc_blk_pos(4*n_blocks);

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        // [TAG_QSA_BIAS] counters for the block bias path of this stream
        int64_t bias_shift = 0, bias_tail_wrong = 0, bias_own_masked = 0, bias_past_bid = 0;
        int64_t bias_first_shift = -1, bias_first_masked = -1, bias_tail_start = -1;
        int64_t ub_n_complete = 0;

        int32_t * cur_cell_blk  = dst_cell_blk + s*n_kv;

        std::fill(loc_blk_cells.begin(), loc_blk_cells.end(), 0);
        std::fill(loc_blk_pos.begin(),   loc_blk_pos.end(),   0);

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                loc_blk_pos[sec*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                loc_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            }

            cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
        }

        if (dst_blk_cells != nullptr) {
            std::copy(loc_blk_cells.begin(), loc_blk_cells.end(), dst_blk_cells + s*(r*n_blocks));
        }

        if (dst_blk_pos != nullptr) {
            for (int64_t sec = 0; sec < 4; ++sec) {
                std::copy(loc_blk_pos.begin() + sec*n_blocks, loc_blk_pos.begin() + (sec + 1)*n_blocks,
                        dst_blk_pos + sec*(n_blocks*n_ns) + s*n_blocks);
            }
        }

        // [TAG_QSA_POOLED_CACHE] resolve which blocks the graph must (re)pool this ubatch:
        // the range from the sequence's watermark to its last complete block. Complete blocks
        // are immutable, so rows below the watermark stay valid; rollbacks arrive as
        // seq_rm/state_read, which clamp the watermark before this runs.
        if (dirty_cells != nullptr) {
            GGML_ASSERT(n_ns == 1 && "the pooled cache path is single-stream only");

            const int64_t n_dirty_max = dirty_rows->ne[0];
            const int64_t row_base    = pooled_row_base(seq_of_stream);   // [TAG_QSA_OWNROW]
            const int64_t dustbin     = row_base + (int64_t) pooled_rows_per_seq - 1;

            int32_t * dst_d_cells = (int32_t *) dirty_cells->data;
            int32_t * dst_d_pos   = (int32_t *) dirty_pos->data;
            int64_t * dst_d_rows  = (int64_t *) dirty_rows->data;

            // the bids are the complete blocks, pushed in position-block order: the last
            // bid's block ends the complete range
            //
            // [TAG_QSA_OWN] only a block this sequence owns may end it. In a unified cache the
            // bid list also carries the complete groups of every other live sequence, and taking
            // the last one let a short request adopt the whole inventory of a long one: it then
            // pooled (and overwrote) rows belonging to that sequence. The bias masks blocks we
            // do not own anyway, so nothing of ours is lost by stopping at our own last block.
            int64_t n_complete = 0;

            for (int32_t t = n_bid - 1; t >= 0; --t) {
                if (one_seq || cells.seq_has((uint32_t) bid_cell[t], seq_of_stream)) {
                    n_complete = (int64_t) bid_idx[t]/r + 1;
                    break;
                }
            }

            auto & w = pooled_valid(seq_of_stream);

            const int64_t w_raw = w; // [TAG_QSA_DBG] watermark before the clamp

            w = std::min(w, n_complete);

            const int64_t n_dirty = n_complete - w;

            // [TAG_QSA_DBG] the ubatch's own position range, and the bound the old
            // position-only sizing would have produced: n_dirty past that bound means the
            // stream-inventory sizing is what kept this ubatch alive.
            int64_t q_lo = -1, q_hi = -1;

            for (int64_t i = 0; i < n_tokens; ++i) {
                const int64_t p = (int64_t) ubatch->pos[i];

                q_lo = q_lo < 0 ? p : std::min(q_lo, p);
                q_hi = q_hi < 0 ? p : std::max(q_hi, p);
            }

            const int64_t old_bound = std::max<int64_t>(1, (q_hi + 1)/r - w);
            const bool    saved     = n_dirty > old_bound;   // [TAG_QSA_SIZING] the fix at work

            // log whenever the requirement reaches the allocated capacity (no headroom left) or
            // the old sizing would have aborted - the ubatch is then captured with full state
            if (n_dirty >= n_dirty_max || saved) {
                const bool fatal = n_dirty > n_dirty_max;
                // steady-state single-token decode is exactly tight (1 == 1) by design; only
                // sample the cases where a non-trivial amount of headroom is fully consumed.
                const bool interesting = (n_dirty == n_dirty_max && n_dirty_max > 1) || saved;
                const bool want  = fatal || (interesting && g_qsa_dbg_spam.fetch_add(1) < 5);

                if (want) {
                    // the sequence's own cells: how far they reach vs the ubatch's positions
                    int64_t n_live = 0, pos_lo = -1, pos_hi = -1;

                    for (int64_t j = 0; j < n_kv; ++j) {
                        if (cells.is_empty((uint32_t) j)) {
                            continue;
                        }

                        const int64_t p = (int64_t) cells.pos_get((uint32_t) j);

                        n_live++;

                        pos_lo = pos_lo < 0 ? p : std::min(pos_lo, p);
                        pos_hi = std::max(pos_hi, p);
                    }

                    // the 2D (M-RoPE) extent, when the ubatch carries one
                    int64_t q_hi_y = -1, q_hi_x = -1;

                    if (ubatch->is_pos_2d()) {
                        for (int64_t i = 0; i < n_tokens; ++i) {
                            q_hi_y = std::max(q_hi_y, (int64_t) ubatch->pos[i + n_tokens]);
                            q_hi_x = std::max(q_hi_x, (int64_t) ubatch->pos[i + n_tokens*2]);
                        }
                    }

                    LLAMA_LOG_ERROR(
                            "%s: [TAG_QSA_DBG] %s n_dirty=%lld n_dirty_max=%lld | fill: n_complete=%lld "
                            "n_bid=%lld w_raw=%lld w=%lld | graph: est_n_dirty_max=%lld est_n_complete=%lld "
                            "est_w=%lld est_q_max=%lld est_ntok=%lld est_calls=%lld est_mock=%lld | "
                            "seq=%d r=%lld n_kv=%lld n_blocks=%lld n_ns=%lld n_tps=%lld n_tokens=%lld "
                            "ranked=%d one_seq=%d pos_2d=%d | cells: n_live=%lld pos_lo=%lld pos_hi=%lld | "
                            "ubatch pos: lo=%lld hi=%lld hi_y=%lld hi_x=%lld | old position-only "
                            "bound=%lld saved_by_stream_sizing=%d\n",
                            __func__,
                            fatal ? "POOLED-CACHE SIZING OVERRUN -> about to abort:" :
                            (saved ? "position-only sizing would have overrun -> covered:" : "no headroom left:"),
                            (long long) n_dirty, (long long) n_dirty_max,
                            (long long) n_complete, (long long) n_bid, (long long) w_raw, (long long) w,
                            (long long) g_qsa_dbg_est.n_dirty_max.load(),
                            (long long) g_qsa_dbg_est.n_complete.load(),
                            (long long) g_qsa_dbg_est.w.load(),
                            (long long) g_qsa_dbg_est.q_max.load(),
                            (long long) g_qsa_dbg_est.n_tokens.load(),
                            (long long) g_qsa_dbg_est.calls.load(),
                            (long long) g_qsa_dbg_est.mock.load(),
                            (int) seq_of_stream, (long long) r, (long long) n_kv, (long long) n_blocks,
                            (long long) n_ns, (long long) n_tps, (long long) n_tokens,
                            (int) ranked, (int) one_seq, (int) ubatch->is_pos_2d(),
                            (long long) n_live, (long long) pos_lo, (long long) pos_hi,
                            (long long) q_lo, (long long) q_hi, (long long) q_hi_y, (long long) q_hi_x,
                            (long long) old_bound, (int) saved);

                    fflush(stderr);

                    g_qsa_dbg_hits.fetch_add(1);
                }
            }

            GGML_ASSERT(n_dirty <= n_dirty_max && "dirty tables sized at graph build; see qsa_pooled_n_dirty_max");

            // position block -> bid: an incomplete block below the complete end pools nothing
            // this time and keeps its stale row, masked by the bias
            std::vector<int32_t> pb_bid(n_complete > 0 ? (size_t) n_complete : 1u, -1);
            for (int32_t t = 0; t < n_bid; ++t) {
                const int64_t pb = bid_idx[t]/r;

                if (pb >= n_complete) {
                    continue;
                }

                // [TAG_QSA_OWN] a block of another sequence is not ours to pool: leave its row
                // to whoever owns it (the bias masks it for us anyway)
                if (!one_seq && !cells.seq_has((uint32_t) bid_cell[t], seq_of_stream)) {
                    continue;
                }

                pb_bid[pb] = t;
            }

            // [TAG_QSA_GUARD] report what the pooled fill does with rows, so that the states the
            // [TAG_QSA_SIZING] inventory sizing let through silently (instead of the old
            // GGML_ASSERT) stay visible:
            //   hole_rows          blocks in [w, n_complete) this ubatch did not bring, or that
            //                      belong to another sequence: written to the dustbin, so the
            //                      row keeps whatever its owner pooled there (expected after
            //                      [TAG_QSA_OWN], and free)
            //   foreign_member     a pooled row whose members belong to another sequence: must
            //                      be 0 since [TAG_QSA_OWN], it is the shape of the old bug
            //   cross_seq          a pooled row overwritten while another sequence's watermark
            //                      still certifies it: rows are position-indexed and shared,
            //                      watermarks are not -> must be 0 since [TAG_QSA_OWN]
            //   foreign_bids       the bid list carried another sequence's blocks (informational)
            // LLAMA_QSA_GUARD=abort makes foreign_member/cross_seq fatal; =0/off disables the
            // reporting (it is ON by default; info lines are capped, the rest by guard_cap).
            const char * guard_env   = getenv("LLAMA_QSA_GUARD");
            const bool   guard_off   = guard_env != nullptr &&
                    (guard_env[0] == '0' || guard_env[0] == 'o' || guard_env[0] == 'O' ||
                     guard_env[0] == 'n' || guard_env[0] == 'N');
            const bool   guard_on    = !guard_off;
            const bool   guard_abort = guard_env != nullptr && (guard_env[0] == 'a' || guard_env[0] == 'A');
            const int64_t guard_cap  = 200;

            int64_t g_holes = 0, g_foreign = 0, g_owner = 0, g_foreign_bids = 0;
            int64_t g_first_hole = -1, g_first_foreign = -1, g_first_owner = -1;

            for (int32_t t = 0; guard_on && t < n_bid; ++t) {
                if (!one_seq && !cells.seq_has((uint32_t) bid_cell[t], seq_of_stream)) {
                    g_foreign_bids++;
                }
            }

            for (int64_t i = 0; i < n_dirty_max; ++i) {
                const bool    live = i < n_dirty;
                const int64_t b    = w + i;
                const int32_t t    = live && b < n_complete ? pb_bid[b] : -1;

                // [TAG_QSA_OWN] a hole is not ours to write: the dustbin holds it and the owner's
                // row keeps its content, instead of being pooled from cell 0 (and certified)
                const bool pooled = live && t >= 0;

                dst_d_rows[i] = pooled ? row_base + b : dustbin;   // [TAG_QSA_OWNROW]

                for (int64_t sec = 0; sec < 4; ++sec) {
                    dst_d_pos[sec*n_dirty_max + i] = pooled ? loc_blk_pos[sec*n_blocks + t] : 0;
                }
                for (int64_t j = 0; j < r; ++j) {
                    dst_d_cells[i*r + j] = pooled ? loc_blk_cells[t*r + j] : 0;
                }

                if (!pooled) {
                    if (guard_on && live) {
                        g_holes++;
                        g_first_hole = g_first_hole < 0 ? b : g_first_hole;
                    }

                    continue;
                }

                if (guard_on) {
                    for (int64_t j = 0; j < r; ++j) {
                        const int32_t c = loc_blk_cells[t*r + j];

                        if (c >= 0 && !cells.seq_has((uint32_t) c, seq_of_stream)) {
                            g_foreign++;
                            g_first_foreign = g_first_foreign < 0 ? b : g_first_foreign;
                            break;
                        }
                    }
                }

                // [TAG_QSA_HEAL] rows are position-keyed and shared while the watermark is per
                // sequence: when this write takes over a row another live sequence still
                // certifies, roll that sequence's watermark back so its next fill re-pools the
                // taken rows instead of scoring keys that belong to us. Memory-free, and it
                // bounds the corruption window to one ubatch of the victim instead of forever.
                const auto ow = pooled_row_owner.find(row_base + b);   // [TAG_QSA_OWNROW]

                if (ow != pooled_row_owner.end() && ow->second != seq_of_stream) {
                    auto wt = pooled_w.find(ow->second);

                    if (wt != pooled_w.end() && wt->second > b) {
                        if (guard_on) {
                            g_owner++;
                            g_first_owner = g_first_owner < 0 ? b : g_first_owner;
                        }

                        wt->second = b;
                    }
                }

                pooled_row_owner[row_base + b] = seq_of_stream;        // [TAG_QSA_OWNROW]
            }

            const bool guard_violation = g_foreign > 0 || g_owner > 0;

            if (guard_on && (guard_violation ||
                    ((g_holes > 0 || g_foreign_bids > 0) && pooled_guard_info < 8)) &&
                pooled_guard_reports < guard_cap) {
                if (guard_violation) {
                    pooled_guard_reports++;
                } else {
                    pooled_guard_info++;
                }

                LLAMA_LOG_ERROR("[TAG_QSA_GUARD]%s seq=%d r=%lld n_kv=%lld n_blocks=%lld n_bid=%d "
                        "one_seq=%d n_tokens=%lld w=%lld n_complete=%lld n_dirty=%lld n_dirty_max=%lld "
                        "| hole_rows=%lld(first=%lld) foreign_member_rows=%lld(first=%lld) "
                        "cross_seq_rows=%lld(first=%lld) foreign_bids=%lld | reports=%lld info=%lld\n",
                        guard_abort ? " ABORT-ON-VIOLATION" : (guard_violation ? " VIOLATION" : " info"),
                        (int) seq_of_stream, (long long) r, (long long) n_kv, (long long) n_blocks,
                        (int) n_bid, (int) one_seq, (long long) n_tokens,
                        (long long) w, (long long) n_complete, (long long) n_dirty,
                        (long long) n_dirty_max,
                        (long long) g_holes, (long long) g_first_hole,
                        (long long) g_foreign, (long long) g_first_foreign,
                        (long long) g_owner, (long long) g_first_owner,
                        (long long) g_foreign_bids,
                        (long long) pooled_guard_reports, (long long) pooled_guard_info);
                fflush(stderr);
            }

            if (guard_abort && guard_violation) {
                GGML_ABORT("[TAG_QSA_GUARD] pooled fill touched another sequence's rows");
            }

            w = n_complete;

            ub_n_complete = n_complete;
        }

        // [TAG_QSA_BIASFIX] the block bias below is indexed by block, but the bid arrays are
        // indexed by bid, and a unified cache with a second sequence interleaves one entry per
        // sequence at the same position block. Build the inverse map once per ubatch, keeping the
        // entry that owns this sequence's cells: an entry at index b then really describes block b.
        std::vector<int32_t> blk_entry(n_blocks, -1);

        for (int32_t t = 0; t < n_bid; ++t) {
            const int64_t pb = bid_idx[t]/r;

            if (pb >= n_blocks || blk_entry[pb] >= 0) {
                continue;
            }

            if (one_seq || cells.seq_has((uint32_t) bid_cell[t], seq_of_stream)) {
                blk_entry[pb] = t;
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            // [TAG_QSA_WINDOW] also keep the last qsa_window_tokens selectable (0 = tail only)
            const int64_t win_start = std::max<int64_t>(0, tail_start - qsa_window_tokens);

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                if (bias_on) {
                    bias_tail_start = tail_start;
                }

                for (int64_t b = 0; b < n_blocks; ++b) {
                    // [TAG_QSA_BIASFIX] the entry that describes block b (not entry b)
                    const int32_t t = blk_entry[b];

                    if (t < 0) {
                        // no complete group of ours at this block: invisible, as before
                        if (bias_on) {
                            bias_past_bid++;   // blocks with no entry of ours

                            if (b < ub_n_complete) {
                                bias_own_masked++;

                                if (bias_first_masked < 0) {
                                    bias_first_masked = b;
                                }
                            }
                        }

                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[t] >= win_start ? 1e9f : 0.0f;

                    // [TAG_QSA_BIAS] verification: with the inverse map these can no longer differ,
                    // so any non-zero count means the map itself is wrong
                    if (bias_on) {
                        if ((int64_t) bid_idx[t]/r != b) {
                            bias_shift++;

                            if (bias_first_shift < 0) {
                                bias_first_shift = b;
                            }
                        }

                        if (((int64_t) bid_idx[t] >= (int64_t) tail_start) != (b*r >= tail_start)) {
                            bias_tail_wrong++;
                        }
                    }
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= win_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }

        // [TAG_QSA_BIAS] report the bias path of this ubatch: only when it disagreed with the block
        // layout (plus one line when the per-cell path takes over, to show which path runs)
        const bool bias_bad = bias_shift > 0 || bias_tail_wrong > 0;
        const bool bias_log = bias_bad || (!blk_bias && pooled_bias_reports < 1);

        if (bias_on && bias_log && pooled_bias_reports < 1000) {
            pooled_bias_reports++;

            LLAMA_LOG_ERROR("[TAG_QSA_BIAS] path=%s seq=%d r=%lld n_kv=%lld n_blocks=%lld n_bid=%d "
                    "one_seq=%d n_tokens=%lld tail_start=%lld n_complete=%lld | shifted_entries=%lld(first=%lld) "
                    "own_complete_masked=%lld(first=%lld) wrong_tail_flag=%lld past_bid_blocks=%lld | reports=%lld\n",
                    blk_bias ? "blk" : "cell",
                    (int) seq_of_stream, (long long) r, (long long) n_kv, (long long) n_blocks,
                    (int) n_bid, (int) one_seq, (long long) n_tokens,
                    (long long) bias_tail_start, (long long) ub_n_complete,
                    (long long) bias_shift, (long long) bias_first_shift,
                    (long long) bias_own_masked, (long long) bias_first_masked,
                    (long long) bias_tail_wrong, (long long) bias_past_bid,
                    (long long) pooled_bias_reports);
            fflush(stderr);
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        ggml_tensor * dirty_cells,
        ggml_tensor * dirty_pos,
        ggml_tensor * dirty_rows) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias,
            dirty_cells, dirty_pos, dirty_rows);
}

ggml_tensor * llama_memory_hybrid_idx_context::get_pooled_k(int32_t il) const {
    return mem != nullptr && get_idx() != nullptr ? mem->get_pooled_k(il) : nullptr;
}

uint32_t llama_memory_hybrid_idx_context::get_pooled_rows() const {
    return mem != nullptr ? mem->get_pooled_rows() : 0;
}

int64_t llama_memory_hybrid_idx_context::pooled_row_base(llama_seq_id seq_id) const {   // [TAG_QSA_OWNROW]
    return mem != nullptr ? mem->pooled_row_base(seq_id) : 0;
}

int64_t llama_memory_hybrid_idx::qsa_stream_idx_max(llama_seq_id seq_id) const {
    const llama_kv_cache * idx_cache = get_mem_idx();

    if (idx_cache == nullptr) {
        return 0;
    }

    const auto & cells = idx_cache->get_cells(seq_id);

    // the fill keys blocks on the cell array: in position space the largest index is the largest
    // position, in ranked (mrope) space it is the cell count - take the larger of the two so
    // neither keying can outrun the tables
    int64_t idx = (int64_t) cells.get_used() - 1;

    for (llama_seq_id sq = 0; sq < LLAMA_MAX_SEQ; ++sq) {
        idx = std::max(idx, (int64_t) cells.seq_pos_max(sq));
    }

    return std::max<int64_t>(idx, 0);
}

uint32_t llama_memory_hybrid_idx_context::qsa_pooled_n_dirty_max(const llama_ubatch & ubatch, uint32_t ratio) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(mem != nullptr);

    // the reserve pass builds worst-case graphs from a mock ubatch with no seq/pos data;
    // give it the per-ubatch bound (the refill after a state load resizes on a live ubatch)
    if (ubatch.seq_id == nullptr || ubatch.seq_id[0] == nullptr || ubatch.pos == nullptr) {
        const uint32_t res = (ubatch.n_tokens + ratio - 1)/ratio + 1;

        qsa_dbg_note(res, -1, -1, -1, (int64_t) ubatch.n_tokens, ratio, -1, true); // [TAG_QSA_DBG]

        return res;
    }

    // single-stream memories only (get_pooled_k gates the callers); like the block tables,
    // the watermark follows the first token's sequence
    const llama_seq_id seq = ubatch.seq_id[0][0];

    llama_pos q_max = -1;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        q_max = std::max(q_max, ubatch.pos[i]);
    }

    // [TAG_QSA_SIZING] the fill counts complete blocks over the stream's whole cell array (in a
    // unified cache that is every sequence sharing it), not over this ubatch's positions: size
    // for that inventory plus the tokens this ubatch adds, or the tables come out short and
    // set_input_qsa aborts. In the steady state both agree and the table stays one row.
    const int64_t idx_max = std::max<int64_t>(mem->qsa_stream_idx_max(seq), (int64_t) q_max) +
                            (int64_t) ubatch.n_tokens;

    // +3 blocks: one for the boundary rounding, two for a slot that adds cells between the
    // graph build and the fill. Rows past the real range are dustbin-padded, so slack costs a
    // few pooled rows, never correctness
    const int64_t n_complete = std::max<int64_t>((int64_t) (q_max + 1)/ratio, idx_max/ratio + 3);
    const int64_t w          = std::min(mem->pooled_valid(seq), n_complete);

    const uint32_t res = (uint32_t) std::max<int64_t>(1, n_complete - w);

    // [TAG_QSA_DBG] what the graph allocated, and the state it was sized from
    qsa_dbg_note(res, n_complete, w, q_max, (int64_t) ubatch.n_tokens, ratio, (int64_t) seq, false);

    return res;
}
