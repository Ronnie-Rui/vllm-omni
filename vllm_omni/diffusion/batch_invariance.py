# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Diffusion batch-invariance switch, seed contract and validated SD3 recipe gates."""

import os
from typing import TYPE_CHECKING

import torch
import vllm.envs as envs

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest

MIN_TORCH_MANUAL_SEED = -(2**63)
MAX_TORCH_MANUAL_SEED = 2**64 - 1

DIFFUSION_BATCH_INVARIANT_ENV = "VLLM_OMNI_DIFFUSION_BATCH_INVARIANT"
_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")

_BIC_SD3_PIPELINE = "StableDiffusion3Pipeline"
_BIC_SD3_ATTENTION_BACKEND = "TORCH_SDPA"
_BIC_SD3_HEIGHT = 512
_BIC_SD3_WIDTH = 512


def diffusion_batch_invariant_enabled() -> bool:
    """Whether diffusion batch invariance is requested.

    Unset (the default) follows vLLM's global ``VLLM_BATCH_INVARIANT`` so mixed
    AR + diffusion pipelines keep a single source of truth. Setting it
    explicitly overrides the global switch in either direction, which lets a
    pipeline enable batch invariance for its LLM stage without forcing the
    diffusion stage into the narrow validated recipe -- and lets non-CUDA
    platforms opt out of the diffusion-side hard requirement.
    """
    raw = os.environ.get(DIFFUSION_BATCH_INVARIANT_ENV)
    if raw is None:
        return bool(envs.VLLM_BATCH_INVARIANT)
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{DIFFUSION_BATCH_INVARIANT_ENV} must be one of {_TRUE_VALUES + _FALSE_VALUES}; got {raw!r}.")


def validate_batch_invariant_diffusion_seed(
    sampling_params: OmniDiffusionSamplingParams,
    *,
    request_id: str,
) -> None:
    """Require a portable diffusion RNG identity in batch-invariant mode."""
    if not diffusion_batch_invariant_enabled():
        return

    if sampling_params.generator is not None:
        raise ValueError(
            "Diffusion batch invariance requires one explicit integer seed and "
            f"does not accept generator input for diffusion request {request_id!r}."
        )

    seed = sampling_params.seed
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            "Diffusion batch invariance requires an explicit integer seed for "
            f"diffusion request {request_id!r}; got {seed!r}."
        )
    if not MIN_TORCH_MANUAL_SEED <= seed <= MAX_TORCH_MANUAL_SEED:
        raise ValueError(
            "Diffusion seed must be in the torch.Generator.manual_seed range "
            f"[{MIN_TORCH_MANUAL_SEED}, {MAX_TORCH_MANUAL_SEED}]; got {seed} "
            f"for request {request_id!r}."
        )


def validate_batch_invariant_sd3_config(od_config: "OmniDiffusionConfig") -> None:
    """Fail closed unless the engine matches the first supported GPU tuple."""
    if not diffusion_batch_invariant_enabled():
        return

    unsupported: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            unsupported.append(message)

    require(
        od_config.model_class_name == _BIC_SD3_PIPELINE,
        f"model_class_name must be {_BIC_SD3_PIPELINE!r}",
    )
    require(od_config.dtype == torch.bfloat16, "dtype must be torch.bfloat16")
    require(bool(od_config.enforce_eager), "enforce_eager must be True")
    # num_gpus == 1 is load-bearing beyond "untested at scale": it backstops the 10
    # DiffusionParallelConfig degrees this gate does not require explicitly (only
    # vae_patch_parallel_size and use_hsdp are), because a single GPU forces every
    # world_size product in DiffusionParallelConfig.__post_init__ to 1 -- and at
    # world_size == 1 GroupCoordinator.all_reduce returns its input untouched, so the
    # diffusion stage performs no collective communication at all.
    require(od_config.num_gpus == 1, "num_gpus must be 1")
    require(od_config.max_num_seqs in {1, 2, 4}, "max_num_seqs must be one of 1, 2, or 4")
    require(not bool(od_config.step_execution), "step_execution must be False")
    require(not bool(od_config.streaming_output), "streaming_output must be False")
    require(od_config.output_type == "latent", "engine output_type must be 'latent'")
    require(od_config.engine_backend == "default", "engine_backend must be 'default'")
    require(
        od_config.distributed_executor_backend == "mp",
        "distributed_executor_backend must be 'mp'",
    )
    require(
        od_config.diffusion_model_runner_cls is None,
        "diffusion_model_runner_cls overrides are not supported",
    )
    # worker_extension_cls is an explicit trusted-code boundary; the evidence
    # harness binds its exact qualname and source hash instead of special-casing it here.
    require(
        od_config.custom_pipeline_args is None,
        "custom_pipeline_args are not supported",
    )
    require(
        od_config.diffusion_load_format == "default",
        "diffusion_load_format must be 'default'",
    )

    parallel_config = od_config.parallel_config
    require(
        parallel_config.vae_patch_parallel_size == 1,
        "parallel_config.vae_patch_parallel_size must be 1",
    )
    require(not bool(parallel_config.use_hsdp), "parallel_config.use_hsdp must be False")

    attention_config = od_config.diffusion_attention_config
    attention_default = attention_config.default
    attention_backend = None if attention_default is None else attention_default.backend
    require(
        isinstance(attention_backend, str) and attention_backend.upper() == _BIC_SD3_ATTENTION_BACKEND,
        f"diffusion attention backend must be explicitly set to {_BIC_SD3_ATTENTION_BACKEND!r}",
    )
    require(not attention_config.per_role, "per-role attention overrides are not supported")

    require(od_config.cache_backend == "none", "cache_backend must be 'none'")
    require(
        not bool(od_config.enable_prompt_embed_cache),
        "prompt embedding cache must be disabled",
    )
    require(od_config.lora_path is None, "LoRA must be disabled")
    require(not bool(od_config.enable_cpu_offload), "CPU offload must be disabled")
    require(
        not bool(od_config.enable_layerwise_offload),
        "layer-wise offload must be disabled",
    )
    require(bool(od_config.vae_use_slicing), "VAE slicing must be enabled")
    require(not bool(od_config.vae_use_tiling), "VAE tiling must be disabled")
    require(od_config.quantization_config is None, "quantization must be disabled")
    require(not od_config.omni_kv_config, "Omni KV transfer must be disabled")

    if unsupported:
        raise ValueError(
            "Diffusion batch invariance currently supports only the validated SD3 single-GPU tuple. "
            "Unsupported configuration: " + "; ".join(unsupported) + "."
        )


def validate_batch_invariant_sd3_request(request: "OmniDiffusionRequest") -> None:
    """Validate external requests against the fixed SD3 evidence recipe."""
    if not diffusion_batch_invariant_enabled():
        return

    params = request.sampling_params
    unsupported: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            unsupported.append(message)

    validate_batch_invariant_diffusion_seed(params, request_id=request.request_id)
    require(
        request.seed_was_explicit,
        "seed must be explicit when OmniDiffusionRequest is constructed",
    )
    require(params.generator_device is None, "generator_device must not be supplied")
    require(params.height == _BIC_SD3_HEIGHT, f"height must be {_BIC_SD3_HEIGHT}")
    require(params.width == _BIC_SD3_WIDTH, f"width must be {_BIC_SD3_WIDTH}")
    # num_inference_steps is deliberately not gated: it is a loop count and changes no
    # tensor shape, so extra steps cannot break the per-step invariance the kernels give.
    # The shape-bearing dimensions above stay gated.
    require(params.num_outputs_per_prompt == 1, "num_outputs_per_prompt must be 1")
    require(
        params.max_sequence_length is None or params.max_sequence_length == 256,
        "max_sequence_length must be None or 256",
    )
    require(params.guidance_scale == 1.0, "guidance_scale must be 1.0")
    require(params.lora_request is None, "request LoRA must be disabled")
    require(params.output_type in ("latent", "pt"), "output_type must be 'latent' or 'pt'")

    if isinstance(request.prompt, dict):
        require(
            set(request.prompt) <= {"prompt", "negative_prompt"},
            "prompt must not contain multimodal or tensor inputs",
        )
        require(isinstance(request.prompt.get("prompt"), str), "prompt['prompt'] must be a string")
    elif not isinstance(request.prompt, str):
        require(False, "only a standard text prompt is supported")

    for field_name in ("latents", "sigmas"):
        require(getattr(params, field_name, None) is None, f"sampling_params.{field_name} must not be supplied")

    if unsupported:
        raise ValueError(
            "Diffusion batch invariance request is outside the validated SD3 recipe. Unsupported request: "
            + "; ".join(unsupported)
            + "."
        )
