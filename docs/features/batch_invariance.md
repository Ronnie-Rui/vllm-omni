# Batch Invariance (Diffusion)

!!! note
    Diffusion batch invariance is experimental. Evidence covers a single Stable Diffusion 3
    configuration on one GPU. Other configurations are **not rejected** — they run, just without
    that evidence.

Batch invariance means a request produces bit-identical output regardless of the batch size it lands
in or the position it occupies inside that batch: the same prompt and seed alone must produce the
same latents as when batched with three other requests.

vLLM-Omni builds on vLLM's batch-invariant kernels. For the autoregressive side see vLLM's own guide;
this page covers the diffusion stage.

## Motivation

- **Reinforcement learning**: RL rollouts must be reproducible, or batching variance enters the reward signal and cannot be separated from real policy change.
- **Debugging**: a bug that only appears at batch size 4 is far easier to isolate when batch size is the only variable that changed.
- **Regression testing**: bit-identical output makes tensor-level assertions possible, so drift is caught by CI instead of by eye.

## Hardware Requirements

Evidence covers exactly one NVIDIA CUDA compute capability: **8.9** (SM89, Ada — an RTX 4090). The
worker does not check compute capability, so any other CUDA GPU is **unverified, not rejected**: the
switch is honoured, the engine runs, and determinism holds for the operators vLLM actually replaces.
Do not read that as "newer is safer" — vLLM *branches* on capability rather than requiring a floor.

**On SM120 (capability 12.0, RTX 5090) batch invariance does not hold**: identical requests diverge
across batch sizes with the switch on. That is observed, not predicted, and the branch explains it —
`enable_batch_invariant_mode()` in `vllm/model_executor/layers/batch_invariant.py` installs the
Triton overrides for `mm`, `addmm`, `matmul` and `linear` only when
`current_platform.is_device_capability_family(80)` holds, i.e. `capability // 10 == 8`. SM89 is in
that family, SM120 is not and takes the cuBLAS-workspace path instead.

ROCm/HIP and non-CUDA devices skip silently: the bootstrap returns before installing anything and
emits no message, so a clean startup there is not evidence of determinism. ROCm is not skipped
because the overrides would be unreachable — `RocmPlatform.dispatch_key` is `"CUDA"`, so upstream's
`softmax`, `_log_softmax`, `_softmax`, `mean.dim` and `bmm` registrations, which sit outside the
`is_cuda()` branch, would land there. What would not land is the GEMM group, gated off by
`is_cuda()`; GEMM is the dominant variance source here, so partial coverage does not make the
property hold. The seed contract, meanwhile, is not device-aware — it follows the switch alone, so an
enabled switch on ROCm still demands an explicit integer seed and still rejects `generator`,
`generator_device`, `latents` and `sigmas` without delivering determinism in return. Set
`VLLM_OMNI_DIFFUSION_BATCH_INVARIANT=0` to release the diffusion stage from a contract it cannot
benefit from; that is also the way out on any device you have not validated.

## Enabling Batch Invariance

| Variable | Effect |
| --- | --- |
| `VLLM_BATCH_INVARIANT` | vLLM's global switch. The diffusion stage follows it by default. |
| `VLLM_OMNI_DIFFUSION_BATCH_INVARIANT` | Diffusion-only override. Unset follows the global switch; `1`/`true`/`yes`/`on` forces on; `0`/`false`/`no`/`off` forces off. |

Leave the diffusion-only variable unset unless the two stages must differ — a mixed AR + diffusion
pipeline may want deterministic text without subjecting diffusion to the [seed
contract](#seed-contract), or may need to opt out where diffusion has no evidence. An unparsable
value raises `ValueError` listing the accepted values rather than defaulting to off: a typo that
silently disabled determinism would be worse than a crash.

```bash
# Whole pipeline deterministic
export VLLM_BATCH_INVARIANT=1

# AR stage deterministic, diffusion stage left alone
export VLLM_BATCH_INVARIANT=1
export VLLM_OMNI_DIFFUSION_BATCH_INVARIANT=0
```

### Seed contract

In batch-invariant mode every diffusion request must carry an explicit integer seed and must not pass
a `generator` object:

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

Without a seed vLLM-Omni assigns a random one so all ranks share an RNG state, so identical requests
legitimately differ and determinism would look broken while the kernels worked correctly; the missing
seed is therefore rejected up front with the request id. A `generator` object is rejected because it
binds to one device and does not travel across worker processes. Three further inputs are rejected
for the same reason, each taking the RNG identity out of the seed's hands:

| Rejected input | Why |
| --- | --- |
| `generator_device` | selects the device the RNG is drawn on, so the seed no longer fixes the draw |
| `latents` | supplies the initial noise directly, so the seed does not determine it at all |
| `sigmas` | replaces the noise schedule, so the seeded trajectory follows a different path |

These are the *only* request-level rejections in batch-invariant mode. They are not recipe
constraints but the premise of "same seed ⇒ same output" itself, which is why they stay while the
configuration checks do not. The check runs from `OmniDiffusionRequest.__post_init__`, immediately
before the random-seed fallback it guards, so it covers every construction path; internal warmup
requests are exempt.

## Tested Configurations

This records **what has been measured**, not what the engine permits:

| Dimension | Validated value |
| --- | --- |
| Pipeline | `StableDiffusion3Pipeline` |
| Attention backend | `TORCH_SDPA` |
| Resolution | 512 × 512 |
| Inference steps | 8 (the only measured value) |
| dtype | `torch.bfloat16` |

Engine settings: single GPU (`num_gpus=1`), `enforce_eager=True`, `output_type="latent"`,
`vae_use_slicing=True`, no caching, LoRA, quantization, offload, VAE patch parallelism, step
execution or streaming. Evidence was collected across batch sizes 1, 2, 3 and 4, three times each,
every comparison pinned to the same physical GPU — batch size 3 comes from `max_num_seqs=4` so a
partial wave forms naturally (capacity mapping `{1: 1, 2: 2, 3: 4, 4: 4}`). To reproduce, run
`StableDiffusion3Pipeline` under those settings with `VLLM_OMNI_DIFFUSION_BATCH_INVARIANT=1` and
compare latents for one seeded prompt across those batch sizes.

**The table is evidence, not a gate.** vLLM-Omni does not reject configurations outside it: a
different pipeline, resolution, dtype, step count or GPU count runs normally, with determinism
holding only for the operators vLLM replaces. Entries outside the table are **unverified, not
unsupported** — they may well be batch-invariant, nobody has checked; please report it if you
validate one. This matches vLLM upstream, which likewise lists validated models and notes that
others may also work. `num_inference_steps` is the one dimension we can reason about rather than
merely leave unmeasured: it is a loop count and changes no tensor shape.

### Multi-GPU is unverified in both directions

Every measurement is single-GPU, with all 12 degrees in `DiffusionParallelConfig` at 1, so multi-GPU
batch invariance is **unverified in both directions** — no evidence either way. It is not gated, so
it runs. The gap is structural: at `world_size == 1` the stage performs no collective communication
at all (`GroupCoordinator.all_reduce` returns its input untouched), so single-GPU runs never exercise
the code multi-GPU determinism would rest on. An audit would have to cover:

- **Upstream has no deterministic collective reduction, only a configuration convergence.**
  `override_envs_for_invariance()` pins NCCL to one channel and the tree all-reduce algorithm
  (`NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS=1`, `NCCL_ALGO=allreduce:tree`, `NCCL_PROTO=Simple`) and
  sets `VLLM_ALLREDUCE_USE_SYMM_MEM=0`. Those are read when a communicator is built, so determinism
  presupposes every worker installs batch invariance first. This bootstrap does run before
  `init_distributed_environment` — but that ordering has never been tested on more than one GPU.
- **`VLLM_ALLREDUCE_USE_SYMM_MEM=0` is unaudited here**: the name appears nowhere in `vllm_omni` and
  is consumed by vLLM's own device communicator, while diffusion calls `torch.distributed.all_reduce`
  directly. Whether the flag reaches this path is unchecked.
- **Which degrees introduce a cross-rank reduction**: `tensor_parallel_size` (row-parallel linear
  all-reduces partial sums), the sequence-parallel degrees `ulysses_degree`/`ring_degree`/
  `allgather_degree`, `vae_patch_parallel_size`, `text_encoder_tp_size`, and HSDP parameter
  all-gathers under `use_hsdp`. `cfg_parallel_size` only all-gathers whole guidance predictions;
  `pipeline_parallel_size` and `data_parallel_size` move no reduction across ranks.
- vLLM sets `disable_custom_all_reduce = True` whenever `VLLM_BATCH_INVARIANT` is on
  (`vllm/config/parallel.py`); that name appears zero times in `vllm_omni`, so the protection does
  not obviously carry over. See vllm#50136 (custom all-reduce eligibility depends on a size
  threshold, making kernel selection a function of batch composition) and vllm#30321 (DP + EP
  inconsistency; fix PR #45683 unmerged and MoE-specific).

### Why only SD3 has evidence

Every other pipeline under `vllm_omni/diffusion/models/` is **unverified, not rejected**. That the
measured one is SD3 follows from operator coverage: **video pipelines add `Conv3d` and audio adds
`Conv1d`**, and vLLM's batch-invariant layer overrides no convolution operator at all, so neither
path is touched by it — video's temporal decode has never been measured either. **`group_norm` is a
separate, image-side gap**, also unoverridden upstream; SD3 is exempt only because the recipe fixes
the shapes its own normalization sees. Extending the evidence waits on upstream batch-invariant
convolution and attention kernels, not on more testing here.

## Implementation Details

1. Resolve the three-state switch during worker startup, after the CUDA device is selected and before
   distributed init, since vLLM writes NCCL variables a live communicator would ignore.
2. Return silently on ROCm/non-CUDA, skipping steps 3-4; capability is not checked, so every CUDA
   device proceeds.
3. Call vLLM's `init_batch_invariance()`, which installs deterministic operators, disables split-k
   and reduced-precision reductions, forces IEEE fp32 for matmul and cuDNN convolution, and pins NCCL
   to deterministic algorithms.
4. Align `VLLM_BATCH_INVARIANT` for the worker process — vLLM re-reads it, and without the alignment
   a diffusion-only opt-in would install nothing.

Expect a throughput cost: deterministic kernels give up optimizations that reorder floating-point
reductions, and that trade is the point of the feature.

## Known Limitations

**The narrow tested configuration follows from operator coverage, not from incomplete testing.**
vLLM's layer overrides matrix multiplication (`mm`, `addmm`, `matmul`, `linear`, `bmm`), softmax
variants and `mean.dim`. It does not override `scaled_dot_product_attention`, `conv2d` or
`group_norm` — exactly what a diffusion pipeline leans on, since `TORCH_SDPA` calls
`scaled_dot_product_attention` directly and the VAE decoder is convolutions plus group
normalization. Their batch invariance is a property of the shapes the validated recipe happens to
produce, not a kernel-layer guarantee: change the resolution and the convolution shapes change with
it, which is why 512 × 512 is measured rather than a supported range. An operator absent from that
list is not thereby non-deterministic — it is unguaranteed, and each case has to be argued
separately. Closing the gap needs those three operators implemented batch-invariantly in vLLM, which
is upstream work. Until then:

- only the configuration above has evidence; the rest runs unverified rather than rejected;
- multi-GPU is unverified — the evidence matrix is single-GPU;
- `torch.compile` paths are unverified; the validated run uses `enforce_eager=True`;
- output beyond `output_type="latent"`/`"pt"` adds VAE decode and PIL encode, which the evidence does
  not cover.

### AudioX is mutually exclusive with the seed contract

`AudioXPipeline` cannot run under batch invariance at all — the one configuration genuinely blocked
rather than merely unverified. The [seed contract](#seed-contract) rejects any request carrying a
`generator`, while `pipeline_audiox.py` raises `"AudioXPipeline requires sampling_params.generator."`
when `generator is None`, so every request is rejected by one side or the other. The contract does not
branch on pipeline; it applies to every request whose stage is of type `diffusion`. Turn batch
invariance off for that stage to run it. The fix belongs on AudioX's `generator` requirement.
