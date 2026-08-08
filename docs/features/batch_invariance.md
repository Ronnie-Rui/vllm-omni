# Batch Invariance (Diffusion)

!!! note
    Diffusion batch invariance is experimental. It currently has evidence for a single
    Stable Diffusion 3 configuration on one GPU. Other configurations are **not
    rejected** — they run, just without that evidence. See
    [Known Limitations](#known-limitations) for the operator gaps behind that narrow scope.

Batch invariance means a request produces bit-identical output regardless of the batch
size it lands in or the position it occupies inside that batch. Send the same prompt and
seed alone, and it must produce the same latents as when it is batched with three other
requests.

vLLM-Omni builds on vLLM's batch-invariant kernels. For the autoregressive side, see
vLLM's own batch invariance guide; this page covers the diffusion stage.

## Motivation

- **Reinforcement learning**: RL training needs reproducible rollouts. If the same
  prompt yields different images depending on how requests happened to batch together,
  that variance enters the reward signal and cannot be separated from real policy change.
- **Framework and model debugging**: a bug that only appears at batch size 4 is far
  easier to isolate when batch size is the only variable that changed.
- **Regression testing**: bit-identical output makes tensor-level assertions possible,
  so numerical drift is caught by CI instead of by eye.

## Hardware Requirements

Evidence covers exactly one NVIDIA CUDA compute capability: **8.9** (SM89, Ada — measured
on an RTX 4090). No other capability has been measured.

The worker does not check compute capability. Any other CUDA GPU is **unverified, not
rejected**: the switch is honoured, the engine runs, and determinism holds only for the
operators vLLM actually replaces. Verifying your own hardware is up to you.

Do not read that as "newer is safer". vLLM *branches* on capability rather than requiring a
floor — in `vllm/model_executor/layers/batch_invariant.py` the Triton persistent-matmul
overrides for `mm`, `addmm`, `matmul` and `linear` are installed on the SM8x family only,
while other families take a different path. A result measured in one capability bracket
therefore says nothing about another.

One negative result is already known: **on SM120 (compute capability 12.0, RTX 5090) batch
invariance does not hold.** That is an observed failure, not a prediction — identical
requests diverge across batch sizes with the switch on. Do not use SM120 for work that
depends on reproducible latents.

ROCm/HIP builds and non-CUDA devices skip silently: no operator replacement happens and no
message is emitted, so the diffusion stage runs with the original operators. vLLM registers
its batch-invariant overrides on the CUDA dispatch key only, so there is nothing to install
there. Because that skip produces no diagnostic, do not infer determinism from a clean
startup on ROCm — batch invariance is simply not in effect.

If a device is not one you have validated, the supported way out is to turn the feature off
(`VLLM_OMNI_DIFFUSION_BATCH_INVARIANT=0`) rather than to assume the kernels carried over.

## Enabling Batch Invariance

Two environment variables control the diffusion stage:

| Variable | Effect |
| --- | --- |
| `VLLM_BATCH_INVARIANT` | vLLM's global switch. The diffusion stage follows it by default. |
| `VLLM_OMNI_DIFFUSION_BATCH_INVARIANT` | Diffusion-only override. Unset follows the global switch; `1`/`true`/`yes`/`on` forces on; `0`/`false`/`no`/`off` forces off. |

Leave the diffusion-only variable unset unless you need the two stages to differ. It
exists because a mixed AR + diffusion pipeline may want deterministic text generation
without subjecting the diffusion stage to the [seed contract](#seed-contract) below — or
may need to opt out on hardware the diffusion side does not support.

An unparsable value raises `ValueError` listing the accepted values, rather than
defaulting to off. A typo that silently disabled determinism would be worse than a crash.

```bash
# Whole pipeline deterministic
export VLLM_BATCH_INVARIANT=1

# AR stage deterministic, diffusion stage left alone
export VLLM_BATCH_INVARIANT=1
export VLLM_OMNI_DIFFUSION_BATCH_INVARIANT=0
```

### Seed contract

In batch-invariant mode every diffusion request must carry an explicit integer seed, and
must not pass a `generator` object:

```python
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

sampling_params = OmniDiffusionSamplingParams(
    seed=1234,          # required: an explicit int in torch.Generator.manual_seed range
    height=512,
    width=512,
    num_inference_steps=8,
    guidance_scale=1.0,
)
```

Without a seed, vLLM-Omni assigns a random one so all ranks share an RNG state — which
means consecutive identical requests legitimately produce different images. Determinism
would appear broken while the kernels were working correctly, so the missing seed is
rejected up front with the request id in the message. A `generator` object is rejected
because it binds to one device and does not travel across worker processes, so ranks
could not be shown to share an RNG identity.

Three further inputs are rejected for the same reason — each would take the RNG identity
out of the seed's hands:

| Rejected input | Why |
| --- | --- |
| `generator_device` | selects the device the RNG is drawn on, so the seed no longer fixes the draw |
| `latents` | supplies the initial noise directly, so the seed does not determine it at all |
| `sigmas` | replaces the noise schedule, so the seeded trajectory follows a different path |

These are the *only* request-level rejections in batch-invariant mode. They are not recipe
constraints — they are the premise of "same seed ⇒ same output" itself, which is why they
stay while the configuration checks do not (see
[The table is evidence, not a gate](#the-table-is-evidence-not-a-gate)).

The check runs in `OmniDiffusionRequest.__post_init__`, immediately before the random-seed
fallback it guards, so it covers every construction path rather than only requests that
enter through the engine. Internal warmup requests are exempt: they are built by the
engine itself and legitimately rely on that fallback.

## Tested Configurations

This section records **what has been measured**, not what the engine permits. Batch
invariance has been verified on exactly one diffusion configuration:

| Dimension | Validated value |
| --- | --- |
| Pipeline | `StableDiffusion3Pipeline` |
| Attention backend | `TORCH_SDPA` |
| Resolution | 512 × 512 |
| Inference steps | 8 (the only measured value) |
| dtype | `torch.bfloat16` |

Engine settings used for that run: single GPU (`num_gpus=1`), `enforce_eager=True`,
`output_type="latent"`, `vae_use_slicing=True`, no caching, no LoRA, no quantization,
no CPU/layer-wise offload, no VAE patch parallelism, no step execution, no streaming.

Evidence was collected across batch sizes 1, 2, 3 and 4, repeated three times, with
every comparison inside a repetition pinned to the same physical GPU. Batch size 3 is
produced by running the engine with `max_num_seqs=4` so a partial wave forms naturally
(the batch-size-to-capacity mapping is `{1: 1, 2: 2, 3: 4, 4: 4}`); reproducing BS3 with
`max_num_seqs=3` exercises a different admission path.

Reproduce with:

```bash
python examples/offline_inference/text_to_image/sd3_batch_invariance_gpu.py
```

### The table is evidence, not a gate

**vLLM-Omni does not reject configurations outside this table.** With batch invariance
enabled, a different pipeline, resolution, dtype, step count or GPU count runs normally —
the engine does not fail closed on any of them. Determinism then holds only for the
operators vLLM actually replaces; anything outside that set (see
[Known Limitations](#known-limitations)) carries no guarantee.

So read the table as *measured*, not as *supported*: entries outside it are **unverified,
not unsupported**. They may well be batch-invariant — nobody has checked. If you validate
another configuration, please report it so the table can grow.

This matches vLLM's own approach upstream, which likewise lists the models it has
explicitly validated and notes that others may also work.

`num_inference_steps` is worth calling out because it is the one dimension we can reason
about rather than merely leave unmeasured: it is a loop count and changes no tensor shape,
so extra steps cannot break the per-step invariance the kernels provide. Only 8 steps have
been measured, but unlike resolution it carries no shape risk.

### Multi-GPU is unverified in both directions

Every measurement above is single-GPU, and so is every parallelism degree behind it: all
12 degrees in `DiffusionParallelConfig` were at 1 for the whole evidence matrix. Multi-GPU
diffusion batch invariance is therefore **unverified in both directions** — there is no
evidence that it holds, and none that it fails. It is not gated, so it runs.

The single-GPU evidence says nothing about the multi-GPU path, and that is a structural gap
rather than a sampling one: on one GPU the diffusion stage performs **no collective
communication at all**. Every `world_size` product in `DiffusionParallelConfig` collapses
to 1, and at `world_size == 1` `GroupCoordinator.all_reduce` returns its input untouched.
Reduction order — the usual source of batch-dependent numerics in collectives — cannot
vary because there is no reduction. So the single-GPU runs never exercise the code whose
determinism multi-GPU determinism would rest on.

### Why only SD3

The first capability check pins `model_class_name` to `StableDiffusion3Pipeline`, so all
other pipelines in `vllm_omni/diffusion/models/` — 43 of them — fail closed. That is an
operator-coverage decision, not a preference for SD3:

- **Video pipelines add `Conv3d`.** `vllm_omni/diffusion` contains 71 `Conv3d` mentions
  (66 of them under `models/`). vLLM's batch-invariant layer overrides none of it: in
  `vllm/model_executor/layers/batch_invariant.py`, both vLLM trees checked here match
  `conv1d|conv3d|convolution|group_norm|scaled_dot_product` zero times.
- **Audio pipelines add `Conv1d`** — 70 mentions under `vllm_omni/diffusion` — with the
  same zero coverage upstream.
- **Video is harder still.** `vllm_omni/diffusion/models/lance/wan_vae.py` defines
  `CausalConv3d` as a "Causal 3D conv with feature-map caching across temporal chunks".
  Carrying cache state across temporal chunks means the per-sample decode that SD3 gets
  from `vae_use_slicing=True` does not translate: a sample's output depends on chunk
  state, not on the sample alone.

Lifting the pipeline gate therefore waits on upstream batch-invariant convolution and
attention kernels, not on more testing here.

### What multi-GPU would need audited

There is an omni-specific gap to check before trusting the multi-GPU path. vLLM sets
`disable_custom_all_reduce = True` whenever `VLLM_BATCH_INVARIANT` is on, in
`vllm/config/parallel.py`. `disable_custom_all_reduce` appears zero times anywhere in
`vllm_omni`, and diffusion calls `torch.distributed.all_reduce` directly rather than
routing through vLLM's `CustomAllreduce`. So that upstream protection does not obviously
carry over to omni diffusion; enabling multi-GPU means auditing it rather than inheriting
it. Two upstream issues describe the failure modes to expect: vllm#50136 (custom
all-reduce eligibility depends on a size threshold, so kernel selection becomes a function
of batch composition) and vllm#30321 (DP + EP inconsistency, whose fix PR #45683 is
unmerged and MoE-specific).

## Implementation Details

When the diffusion stage runs batch-invariant, vLLM-Omni:

1. resolves the three-state switch during worker startup, after the CUDA device is
   selected and before distributed initialization — the ordering matters because vLLM's
   initialization writes NCCL environment variables that a live communicator would ignore;
2. returns silently on ROCm/non-CUDA devices, so steps 3 and 4 are skipped entirely there;
   compute capability is not checked, so every CUDA device proceeds;
3. calls vLLM's `init_batch_invariance()`, which installs deterministic operator
   implementations, disables split-k and reduced-precision reductions, forces IEEE fp32
   precision for matmul and cuDNN convolution, and pins NCCL to deterministic algorithms;
4. aligns `VLLM_BATCH_INVARIANT` for the worker process, because vLLM re-reads that
   variable itself — without the alignment, a diffusion-only opt-in would install nothing.

Expect a throughput cost. Deterministic kernels give up optimizations that reorder
floating-point reductions; that trade is the point of the feature.

## Known Limitations

**The narrow tested configuration is a consequence of operator coverage, not of
incomplete testing.** vLLM's batch-invariant layer overrides matrix multiplication
(`mm`, `addmm`, `matmul`, `linear`, `bmm`), softmax variants and `mean.dim`. It does not
override `scaled_dot_product_attention`, `conv2d`, or `group_norm`.

Those three are exactly what a diffusion pipeline leans on: the `TORCH_SDPA` backend
calls `torch.nn.functional.scaled_dot_product_attention` directly, and the VAE decoder is
built from convolutions and group normalization. Their batch invariance is therefore not
guaranteed by the kernel layer — it is a property of the specific shapes the validated
recipe happens to produce. Change the resolution and the convolution shapes change with
it, which is why 512 × 512 is listed as measured rather than as a supported range.

Closing the gap requires batch-invariant implementations of those three operators in
vLLM, which is upstream work. Until then:

- only the configuration in [Tested Configurations](#tested-configurations) has evidence,
  and the rest runs unverified rather than being rejected;
- multi-GPU diffusion batch invariance is unverified; the evidence matrix is single-GPU;
- `torch.compile` paths are unverified — the validated run uses `enforce_eager=True`;
- image output beyond `output_type="latent"`/`"pt"` adds a VAE decode and PIL encode step
  that the evidence does not cover.

### AudioX is mutually exclusive with the seed contract

`AudioXPipeline` cannot run under batch invariance at all — the one configuration that is
genuinely blocked rather than merely unverified. The [seed contract](#seed-contract)
rejects any request carrying a `generator` object, while `pipeline_audiox.py` raises
`"AudioXPipeline requires sampling_params.generator."` when `generator is None`. Every
request is rejected by one side or the other.

The seed contract does not branch on pipeline: it applies to every request whose stage is
of type `diffusion`. So an AudioX request under batch invariance fails either way; turn
batch invariance off for that stage (`VLLM_OMNI_DIFFUSION_BATCH_INVARIANT=0`) to run it.
The fix belongs on AudioX's `generator` requirement rather than on the contract.
