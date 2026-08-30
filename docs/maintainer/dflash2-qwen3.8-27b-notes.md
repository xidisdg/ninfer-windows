# DFlash2 for Qwen3.8-27B — research notes (WIP)

Branch: `dflash-3.8-27b`. Goal: add DFlash2 support to the `qwen3.8-27b` target
(shared with `qwen3.6-27b` via `registry.cpp`; the 27B target now defines the full
`DFlashConfig` v2 geometry; the DFlash2 v2 decode/verify runtime is enabled).

## 0. Status (validate-first increment)

- Converter: **done and verified**. `tools/convert/qwen3_8_27b/convert_dflash2.py` (profile
  `groupwise-int-dflash2`, `weights_id = groupwise-int-dflash2`) ran end-to-end on the
  official Qwen3.8-27B + `z-lab/Qwen3.8-27B-DFlash2` checkpoints: 1190 objects (1184
  tensors + 6 resources), 20255474176 artifact bytes; all 66 DFlash2 recipes materialized
  and W8/BF16-encoded.
- Engine (load path): **done**. `DFlashConfig` v2 geometry is in the 27B `config.h`; the
  `Qwen38GroupwiseIntDflash2` weights profile loads, binds, and shape-validates the
  `dflash2/` component (see `qwen3.8-27b-artifact.md` Section 14). Existing `groupwise-int`
  and `nvfp4` artifacts are unaffected (the dflash2 bind is profile-gated).
- Engine (runtime): **done and enabled** (this increment). `DFlashConfig::supported` is `true`
  and `kMaximumDFlashDraftTokens` is 7 (block size 8 minus 1), so `--spec dflash` starts the v2
  drafter for this target: two-tap dynamic conv + all-sliding-window (window 2048) draft
  attention, the on-device candidate selector lattice build (top-16 + codebooks + pad-to-5120
  rows), and the host lattice path trace (seed ^ 0x85ebca6b, greedy argmax chain or seeded
  sampling with sparse per-position distributions). The v2 decode round runs eager (no CUDA
  graph capture; the drafter owns its all-local 2048-window cache, not the target KV). A
  DFlash draft window outside [1, 7] is rejected at startup.
  The v2 full-block geometry (draft window forced to block size minus 1, CUDA graphs  disabled) applies only while the DFlash backend is selected; `--spec mtp` keeps the  requested MTP draft window in [1, 5] and the usual MTP graph policy, so the two  speculative modes coexist on this target.
- Note: HF republished the Qwen3.8 frontend files on 2026-08-13 (chat template rewritten,
  `model_max_length` 131072 -> 262144); a fresh download fails the pinned-hash preflight.
  The verified conversion used the pinned frontend bytes recovered from the existing
  `qwen3_8_27b.ninfer` artifact. Re-pinning is a conscious decision if converting from a
  fresh download.


## 1. Inputs required

- Base / target: `Qwen/Qwen3.8-27B` (already a fixed source of the existing
  converters — `tools/convert/qwen3_8_27b/convert.py` and `convert_nvfp4.py`).
- Draft model: `z-lab/Qwen3.8-27B-DFlash2` (mirror of `incoai/Qwen3.8-27B-DFlash2`),
  single `model.safetensors` (~3.85 GB, all BF16), arch `DFlash2DraftModel`.
  llama.cpp PR 27342 registers this arch on the existing Qwen3-based DFlashModel
  converter class, so the HF layout is the DFlash1 layout plus DFlash2 extras.

### Draft model config (from HF `config.json`)

| Fact | Value |
|---|---:|
| layers | 5, all `sliding_attention`, window 2048 |
| hidden | 5120 |
| intermediate (dense MLP, no MoE) | 17408 |
| q / kv heads, head dim | 32 / 8, 128 (q 4096, k 1024, v 1024) |
| block size | 8 (7 draft tokens per verify step) |
| mask token id | 248070 (an *existing* added token, not a dedicated row) |
| target layer ids | `[5, 19, 33, 47, 61]` → `fc` input width `5 × 5120 = 25600` |
| non-causal draft attention | `is_causal: false` |
| rope theta | 1e7 (matches target) |

### DFlash2 vs DFlash1 deltas (from llama.cpp commit 5ecbe1ac)

1. **Two-tap dynamic convolutions** on every layer, on the attention input and
   output and the MLP input and output:
   - `layers.{i}.attention_conv.base_kernel`  `[2,2,5120]` = `[side, tap, channel]`
   - `layers.{i}.attention_conv.kernel_projection` `[1280,5120]` =
     `[n_groups=5120/16, kernel=2, side=2]` rows per column; coefficients are a
     low-rank projection of the *normalized* hidden state
     (`build_lora_mm(kernel_projection, normed_hidden)`).
   - `layers.{i}.mlp_conv.*` — same shapes; coefficients from the post-norm
     hidden entering the MLP.
   - Semantics (llama.cpp `build_dflash2_conv`): per group of 16 channels,
     per tap, per side: `weight[ch] = base[tap, side, ch] + coeff[group, tap,
     side]`; output = Σ_tap `weight ⊗ shift_by_tap(block_input)` where the
     shift is *within the block* (block_size 8) with zero padding at the block
     front. Two sides: side 0 applies to the normed input, side 1 to the layer
     output (attention/MLP respectively).
2. **Candidate selector** (replaces DFlash1's per-position top-10 draft
   sampling):
   - `candidate_selector.predecessor_codebook` `[248320, 256]` (full vocab, rank 256)
   - `candidate_selector.successor_codebook`  `[248320, 256]`
   - `candidate_selector.hidden_projection`   `[256, 5120]`
   - The draft graph, after the final norm and the (aliased, target) output
     head, takes `top_k = 16` per position and packs per-position lattice rows
     of width `n_embd = 5120`:
     `[candidate_ids as f32 (16), scores (16 × 16), zeros pad]`.
     `scores[cand, pred] = successor[cand] · (predecessor[pred] ⊙
     hidden_proj(h_pos)) + unary_logit(cand)`; position 1's predecessor is the
     anchor token, later positions' predecessor is the *candidate set* of the
     previous position (broadcast for pos 1).
   - The host then traces one path through the lattice: seeded RNG (seed =
     request seed ^ 0x85ebca6b), per position sample/argmax over
     `row[top_k + pred*top_k : +top_k]`; greedy (temperature ≤ 0) → argmax →
     lossless vs target.
   - Consequences: the draft never consumes raw per-position logits
     (`set_embeddings_nextn(ctx, true, masked=false)`, draft batch rows get
     `logits=false`); the draft model has **no private output head** — the
     target `text/output_head` is aliased, same as DFlash1.
3. **Optional scalars** (`logit_scale`, `final_logit_softcapping`,
   `input_embedding_scale`): all absent/zero in this checkpoint — keep as
   optional metadata, do not require.
4. Encoder (context K/V injection) is structurally the same as DFlash1:
   concatenated target-layer-input features → `fc [5120,25600]` →
   `hidden_norm` → draft context K/V cache. DFlash1's `context_norm` maps to
   the checkpoint's `hidden_norm`.
5. Layer tensor set per draft layer (BF16 in the checkpoint):
   `input_layernorm [5120]`, `q_proj [4096,5120]`, `k_proj [1024,5120]`,
   `v_proj [1024,5120]`, `q_norm [128]`, `k_norm [128]`, `o_proj [5120,4096]`,
   `post_attention_layernorm [5120]`, `gate_proj [17408,5120]`,
   `up_proj [17408,5120]`, `down_proj [5120,17408]`,
   `attention_conv.base_kernel [2,2,5120]`,
   `attention_conv.kernel_projection [1280,5120]`,
   `mlp_conv.base_kernel [2,2,5120]`,
   `mlp_conv.kernel_projection [1280,5120]` — 14 objects/layer × 5 layers,
   plus globals `fc [5120,25600]`, `hidden_norm [5120]`, `norm [5120]`, and the
   three selector objects: 73 named tensors (81 entries incl. the `model.`
   prefix handled by mapping).

## 2. Design decisions (1-4 resolved; 5-7 track the pending runtime increment)

Decisions 1-4 carry their resolutions inline below; items 5-7 list the
pending work and are the plan of record for the runtime increment.


1. **Artifact identity.** 35B precedent: DFlash is baked into the one complete
   product image (no second artifact/profile). 3.8 has *two* registered
   profiles (`groupwise-int`, `nvfp4`). Do both gain a `dflash2/` component, or
   does only one profile carry it? (Converter + docs + binder must agree.)
   **Resolved:** only the `groupwise-int` family carries the drafter, as the new
   `groupwise-int-dflash2` profile (`nvfp4` stays drafter-free); `dflash2/` objects bind
   only under `Qwen38GroupwiseIntDflash2`, so existing artifacts are unaffected.
2. **Namespace / naming.** `dflash/` is taken by the v1 contract on 35B. Suggest
   `dflash2/` namespace for the 27B target to keep v1 and v2 contracts separate
   even though a given target only ever ships one of them.
   **Resolved:** `dflash2/` namespace on the 27B target, separate from the 35B `dflash/`
   v1 contract.
3. **Codebook placement.** The two codebooks are ~127 MB BF16 each. The host
   selector trace only needs 16 rows of 256-dim vectors per step —
   `ValidateOnly`/host-retained (pinned) is enough and saves ~254 MB VRAM.
   The lattice build (top_k + hidden projection) happens on-device in the
   draft graph; only the path trace moves host-side.
   **Resolved:** the codebooks are W8-encoded device tensors in the artifact; the host path
   trace consumes only the lattice rows the device pack produces, so no pinned codebook
   copy is needed at runtime.
4. **Backend flag.** Keep `--spec dflash`; let the target's config select v1
   vs v2 semantics (35B → v1, 3.8-27B → v2). Draft-token cap for v2 is
   `block_size - 1 = 7` (v1 stays 15). Needs a per-target max in
   `speculative_options` validation.
   **Resolved:** `--spec dflash` is kept; the target selects v1 vs v2 semantics. The v2
   draft cap is `block_size - 1 = 7`, wired through `kMaximumDFlashDraftTokens` (0 in the
   validate-first increment, 7 with the runtime).
5. **Engine pieces to add (27B target):**
   - `DFlashConfig` v2 facts (5 sliding layers, window 2048, hidden 5120,
     intermediate 17408, q/kv/head, conv kernel 2 / group 16, selector rank
     256 / top_k 16, block 8, feature layers [5,19,33,47,61]).
   - Persistent layout: all-local cyclic KV (no full-context layer — different
     from 35B v1 which had one).
   - Ops: two-tap dynamic conv (block-local shift with zero pad, per-group
     weight = base + lowrank coeff), non-causal draft attention, lattice build
     (top_k + codebook matmul + pad to 5120 row).
   - Host: lattice path trace (seeded, per request seed), acceptance/verify
     against the target block (target machinery already exists for dflash v1).
   - Target side: feature extraction at layer *inputs* 5/19/33/47/61 (the v1
     feature-append path generalizes; 5 features instead of 8).
   - Graph profiles: v2 `dflash_graph_profiles` for the 27B variant (the stub
     currently returns `{}`).
6. **Converter pieces:** preflight of the DFlash2 checkpoint (config + tensor
   inventory + shapes), `dflash2/` object writer (BF16 → `W8G32_F16S` for the
   big matrices? 35B v1 stored DFlash matrices as `W8G32_F16S`/BF16 — decide
   for v2: matrices `[4096|1024,5120]`, `[5120,4096]`, `[17408,5120]`,
   `[5120,17408]`, `[1280,5120]`, `[5120,25600]`; codebooks `[248320,256]`),
   mask-token embedding is *alias-only* (row 248070 of `text/token_embedding`),
   docs: new `docs/maintainer/qwen3.8-27b-dflash2-artifact.md` section.
7. **Verification:** a reference implementation to diff against — candidates:
   the z-lab/dflash reference (blog links sglang/vllm engines), or port the
   llama.cpp `speculative.cpp` dflash path as a Python reference harness, or
   compare acceptance/quality end-to-end via serve. `tools/reference/` has no
   27B dflash reference yet (35B has mtp/dflash refs).

## 3. Reference material (local)

- `C:\Users\natpa\dflash2_ref\llama_pr27342_commit.patch` — full llama.cpp diff
  (dflash.cpp graph + conv + lattice, speculative.cpp draft/verify, conversion
  mapping, GGUF keys).
- `C:\Users\natpa\dflash2_ref\config.json`, `README.md` (eval tables: DFlash2
  beats MTP acceptance length on all five tasks), `hf_model_info.json`,
  `header_chunk.bin` (safetensors header dump, 81 tensors).

## 4. Suggested sequencing

1. Converter + artifact contract + docs (produces a buildable `.ninfer`),
   bind-only / validate-first so the artifact can be inspected before engine
   work.
2. Engine: layout + ops + round state + host selector + verify; `--spec dflash`
   on the 3.8 target.
3. Reference harness + numeric diff (draft logits vs z-lab/llama.cpp reference),
   then live serve verification with acceptance stats.