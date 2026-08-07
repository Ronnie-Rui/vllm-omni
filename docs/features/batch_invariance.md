# Batch Invariance (Diffusion)

!!! note
    Diffusion batch invariance is experimental. It currently has evidence for a single
    Stable Diffusion 3 configuration on one GPU. See
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

Diffusion batch invariance requires an NVIDIA CUDA GPU with compute capability 8.0
(SM80, Ampere) or newer. This matches vLLM's requirement.

These are enforced in code and raise `RuntimeError` at worker startup:

- non-CUDA devices are rejected;
- ROCm/HIP builds are rejected, because vLLM registers its batch-invariant operator
  overrides on the CUDA dispatch key only — on ROCm the switch would appear to work
  while the original operators kept running;
- compute capability below 8.0 is rejected.

The last one deserves a word. vLLM does not merely skip unsupported hardware: it
*branches* on capability, installing Triton persistent-matmul overrides on SM80 and
relying on cuBLAS workspace configuration on SM90/SM100. Hardware below SM80 falls into
the newer-GPU branch, whose assumptions do not hold there. Failing at startup is
deliberate — a silent fallback would return plausible images with no determinism.

## Enabling Batch Invariance

Two environment variables control the diffusion stage:

| Variable | Effect |
| --- | --- |
| `VLLM_BATCH_INVARIANT` | vLLM's global switch. The diffusion stage follows it by default. |
| `VLLM_OMNI_DIFFUSION_BATCH_INVARIANT` | Diffusion-only override. Unset follows the global switch; `1`/`true`/`yes`/`on` forces on; `0`/`false`/`no`/`off` forces off. |

Leave the diffusion-only variable unset unless you need the two stages to differ. It
exists because a mixed AR + diffusion pipeline may want deterministic text generation
without pinning the diffusion stage to the narrow validated recipe below — or may need
to opt out on hardware the diffusion side rejects.

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

The check runs in `OmniDiffusionRequest.__post_init__`, immediately before the random-seed
fallback it guards, so it covers every construction path rather than only requests that
enter through the engine. Internal warmup requests are exempt: they are built by the
engine itself and legitimately rely on that fallback.

## Tested Configurations

Batch invariance has been verified on exactly one diffusion configuration:

| Dimension | Validated value |
| --- | --- |
| Pipeline | `StableDiffusion3Pipeline` |
| Attention backend | `TORCH_SDPA` |
| Resolution | 512 × 512 |
| Inference steps | 8 (the only measured value; other step counts are not gated) |
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

While batch invariance is enabled, the engine fails closed on anything outside this
tuple, **with one exception — the step count**: `DiffusionEngine.__init__` raises
`ValueError` listing every unsupported engine setting, and `add_request` does the same for
per-request parameters such as pipeline, attention backend and resolution. Those
dimensions determine convolution and attention shapes, so a change there can silently
break invariance; the gate reports them as unsupported rather than returning
non-deterministic output.

`num_inference_steps` is deliberately not gated. It is a loop count and changes no tensor
shape, so extra steps cannot break the per-step invariance the kernels provide. Only 8
steps have been measured, so any other count is unverified rather than rejected — it runs,
and verifying it is up to you. Other resolutions and pipelines may well be
batch-invariant too; they are simply unverified *and* still gated, because there the
failure would be silent. If you validate another configuration, please report it so this
table and the gate can grow together.

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

### Why the single-GPU gate is load-bearing

`num_gpus must be 1` is not merely "multi-GPU is untested". It is the check that keeps the
whole parallelism surface out of scope:

- `DiffusionParallelConfig` exposes 12 parallelism degrees. The gate requires only two of
  them explicitly (`vae_patch_parallel_size == 1`, `use_hsdp == False`); the remaining ten
  are held at 1 indirectly, because `__post_init__` derives `world_size` from the product
  of `pipeline_parallel_size`, `data_parallel_size`, `tensor_parallel_size`,
  `ulysses_degree`, `ring_degree` and `cfg_parallel_size`, and a single GPU forces that
  product to 1. `vae_patch_parallel_size` and `enable_expert_parallel` are not in that
  product — the former is why it carries its own `require`, and the latter is harmless
  today because its degree is `tp × dp` (both 1 on one GPU) and SD3 is not an MoE model.
- With `world_size == 1`, `GroupCoordinator.all_reduce` returns its input immediately, so
  the diffusion stage issues no collective communication at all. Reduction order — the
  usual source of batch-dependent numerics in collectives — cannot vary because there is
  no reduction.

There is also an omni-specific gap waiting on the other side of that gate. vLLM sets
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
2. checks the hardware requirements above and fails closed;
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
  and the engine rejects the rest while batch invariance is on — except the step count,
  which is not gated because it carries no shape, so step counts other than 8 run
  unverified;
- multi-GPU diffusion batch invariance is unverified; the evidence matrix is single-GPU;
- `torch.compile` paths are unverified — the validated run uses `enforce_eager=True`;
- image output beyond `output_type="latent"`/`"pt"` adds a VAE decode and PIL encode step
  that the evidence does not cover.

### AudioX is mutually exclusive with the seed contract

`AudioXPipeline` cannot run under batch invariance at all, for a reason independent of the
capability gate. The [seed contract](#seed-contract) rejects any request carrying a
`generator` object, while `pipeline_audiox.py` raises
`"AudioXPipeline requires sampling_params.generator."` when `generator is None`. Every
request is rejected by one side or the other.

The seed contract does not branch on pipeline: it applies to every request whose stage is
of type `diffusion`. This costs nothing today, because the capability gate refuses AudioX
before the contract is reached — but it is the first thing to resolve when audio pipelines
are opened up, and the fix belongs on AudioX's `generator` requirement rather than on the
contract.
