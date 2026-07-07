"""KV cache efficiency metrics for vLLM-Omni.

This module owns both the pure formulas and the Prometheus observe API. Runtime
hooks feed it with allocator-backed block IDs when available, and stage-token
fallbacks otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from vllm_omni.metrics import definitions as defs

_labelnames = list(defs.KV_LABELS)
_kv_families: dict[str, Any] | None = None


@dataclass(frozen=True)
class KVEfficiencySnapshot:
    """One logical KV allocation snapshot for a stage request."""

    block_size: int
    sequence_tokens: int
    allocated_blocks: int
    allocated_tokens: int
    occupied_tokens: int
    tail_waste_tokens: int
    occupancy_ratio: float
    fragmentation_ratio: float
    footprint_tokens: int
    footprint_bytes: int
    bytes_per_token: int
    cached_tokens: int = 0
    prompt_tokens: int = 0
    prefix_hit_ratio: float = 0.0
    source: str = "stage_tokens"


def _coerce_positive_int(value: Any, default: int = 0) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced if coerced > 0 else default


def normalize_kv_block_ids(raw_block_ids: Any) -> list[int]:
    """Normalize vLLM KV block id shapes to the first cache group's flat list."""

    if raw_block_ids is None:
        return []
    if isinstance(raw_block_ids, tuple) and raw_block_ids and isinstance(raw_block_ids[0], (list, tuple)):
        raw_block_ids = raw_block_ids[0]
    if isinstance(raw_block_ids, list) and raw_block_ids and isinstance(raw_block_ids[0], (list, tuple)):
        raw_block_ids = raw_block_ids[0]
    try:
        return [int(block_id) for block_id in raw_block_ids]
    except (TypeError, ValueError):
        return []


def _cache_dtype_nbytes(cache_dtype: Any, fallback_dtype: Any = None) -> int:
    """Best-effort byte width for vLLM cache dtypes."""

    if cache_dtype is not None and str(cache_dtype).lower() in {"auto", "none"}:
        cache_dtype = None
    value = str(cache_dtype or fallback_dtype or "").lower()
    if not value:
        return 0
    if "fp8" in value or "float8" in value or "int8" in value:
        return 1
    if "bf16" in value or "bfloat16" in value or "fp16" in value or "float16" in value or "half" in value:
        return 2
    if "fp32" in value or "float32" in value:
        return 4
    return 0


def _first_positive_attr(source: Any, names: tuple[str, ...]) -> int:
    for name in names:
        value = getattr(source, name, None)
        coerced = _coerce_positive_int(value)
        if coerced > 0:
            return coerced
    return 0


def estimate_kv_bytes_per_token(vllm_config: Any | None) -> int:
    """Estimate per-token KV bytes from a stage VllmConfig.

    Formula: layers * 2(K+V) * num_kv_heads * head_dim * dtype_bytes.
    Returns 0 when the required model/cache attributes are unavailable.
    """

    if vllm_config is None:
        return 0
    model_config = getattr(vllm_config, "model_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if model_config is None:
        return 0

    get_num_layers = getattr(model_config, "get_num_layers", None)
    get_num_kv_heads = getattr(model_config, "get_num_kv_heads", None)
    get_head_size = getattr(model_config, "get_head_size", None)
    try:
        num_layers = int(get_num_layers(parallel_config) if callable(get_num_layers) else 0)
    except TypeError:
        try:
            num_layers = int(get_num_layers() if callable(get_num_layers) else 0)
        except Exception:
            num_layers = 0
    except Exception:
        num_layers = 0

    try:
        num_kv_heads = int(get_num_kv_heads(parallel_config) if callable(get_num_kv_heads) else 0)
    except TypeError:
        try:
            num_kv_heads = int(get_num_kv_heads() if callable(get_num_kv_heads) else 0)
        except Exception:
            num_kv_heads = 0
    except Exception:
        num_kv_heads = 0

    try:
        head_dim = int(get_head_size() if callable(get_head_size) else 0)
    except Exception:
        head_dim = 0

    if num_layers <= 0:
        num_layers = _first_positive_attr(model_config, ("num_hidden_layers", "num_layers", "n_layer"))
    if num_kv_heads <= 0:
        num_kv_heads = _first_positive_attr(
            model_config,
            ("num_key_value_heads", "num_kv_heads", "num_attention_heads", "n_head"),
        )
    if head_dim <= 0:
        head_dim = _first_positive_attr(model_config, ("head_dim",))
    if head_dim <= 0:
        hidden_size = _first_positive_attr(model_config, ("hidden_size",))
        num_attention_heads = _first_positive_attr(model_config, ("num_attention_heads", "n_head"))
        if hidden_size > 0 and num_attention_heads > 0:
            head_dim = hidden_size // num_attention_heads

    cache_dtype = getattr(cache_config, "cache_dtype", None) if cache_config is not None else None
    fallback_dtype = getattr(model_config, "dtype", None)
    dtype_bytes = _cache_dtype_nbytes(cache_dtype, fallback_dtype)

    if min(num_layers, num_kv_heads, head_dim, dtype_bytes) <= 0:
        return 0
    return int(num_layers * 2 * num_kv_heads * head_dim * dtype_bytes)


def resolve_kv_block_size(vllm_config: Any | None, default: int = 16) -> int:
    """Resolve stage KV block size from VllmConfig-like objects."""

    if vllm_config is not None:
        cache_config = getattr(vllm_config, "cache_config", None)
        block_size = _coerce_positive_int(getattr(cache_config, "block_size", None))
        if block_size > 0:
            return block_size
        block_size = _coerce_positive_int(getattr(vllm_config, "block_size", None))
        if block_size > 0:
            return block_size
    return int(default)


def compute_kv_efficiency(
    *,
    block_size: int,
    sequence_tokens: int,
    allocated_blocks: int | None = None,
    bytes_per_token: int = 0,
    cached_tokens: int = 0,
    prompt_tokens: int | None = None,
    source: str = "stage_tokens",
) -> KVEfficiencySnapshot:
    """Compute logical KV block occupancy and tail waste.

    ``sequence_tokens`` is the number of useful tokens occupying KV blocks for
    this snapshot. ``allocated_blocks`` may come from a real allocator; if it is
    absent, the function uses ``ceil(sequence_tokens / block_size)``.
    """

    block_size = _coerce_positive_int(block_size, default=16)
    sequence_tokens = max(int(sequence_tokens or 0), 0)
    min_blocks = math.ceil(sequence_tokens / block_size) if sequence_tokens > 0 else 0
    if allocated_blocks is None:
        allocated_blocks = min_blocks
    allocated_blocks = max(int(allocated_blocks or 0), min_blocks)
    allocated_tokens = allocated_blocks * block_size
    occupied_tokens = min(sequence_tokens, allocated_tokens)
    tail_waste_tokens = max(allocated_tokens - occupied_tokens, 0)
    occupancy_ratio = (occupied_tokens / allocated_tokens) if allocated_tokens > 0 else 0.0
    fragmentation_ratio = (tail_waste_tokens / allocated_tokens) if allocated_tokens > 0 else 0.0
    bytes_per_token = max(int(bytes_per_token or 0), 0)
    footprint_bytes = allocated_tokens * bytes_per_token
    cached_tokens = max(int(cached_tokens or 0), 0)
    prompt_tokens = sequence_tokens if prompt_tokens is None else max(int(prompt_tokens or 0), 0)
    prefix_hit_ratio = (cached_tokens / prompt_tokens) if prompt_tokens > 0 else 0.0

    return KVEfficiencySnapshot(
        block_size=block_size,
        sequence_tokens=sequence_tokens,
        allocated_blocks=allocated_blocks,
        allocated_tokens=allocated_tokens,
        occupied_tokens=occupied_tokens,
        tail_waste_tokens=tail_waste_tokens,
        occupancy_ratio=occupancy_ratio,
        fragmentation_ratio=fragmentation_ratio,
        footprint_tokens=allocated_tokens,
        footprint_bytes=footprint_bytes,
        bytes_per_token=bytes_per_token,
        cached_tokens=cached_tokens,
        prompt_tokens=prompt_tokens,
        prefix_hit_ratio=prefix_hit_ratio,
        source=source,
    )


def _families() -> dict[str, Any]:
    """Return Prometheus families, creating them after vLLM metric reset.

    vLLM's PrometheusStatLogger unregisters all collectors whose name contains
    "vllm" during startup. Import-time custom collectors would be removed
    before the first request, so vLLM-Omni families are initialized lazily when
    a request actually observes a snapshot.
    """

    global _kv_families
    if _kv_families is not None:
        try:
            from prometheus_client import REGISTRY

            registered = getattr(REGISTRY, "_collector_to_names", {})
            if all(family in registered for family in _kv_families.values()):
                return _kv_families
            for family in _kv_families.values():
                if family in registered:
                    REGISTRY.unregister(family)
        except Exception:
            return _kv_families
        _kv_families = None

    from prometheus_client import Gauge, Histogram

    _kv_families = {
        "footprint_tokens": Gauge(
            defs.KV_FOOTPRINT_TOKENS,
            "Logical allocated KV block capacity in tokens for the latest stage snapshot.",
            labelnames=_labelnames,
        ),
        "footprint_bytes": Gauge(
            defs.KV_FOOTPRINT_BYTES,
            "Estimated allocated KV footprint in bytes for the latest stage snapshot.",
            labelnames=_labelnames,
        ),
        "block_occupancy": Histogram(
            defs.KV_BLOCK_OCCUPANCY_RATIO,
            "Per-stage KV block occupancy ratio: useful tokens / allocated block capacity.",
            labelnames=_labelnames,
            buckets=defs.RATIO_BUCKETS,
        ),
        "tail_waste": Histogram(
            defs.KV_TAIL_WASTE_TOKENS,
            "Per-stage KV tail waste in tokens: allocated block capacity minus useful tokens.",
            labelnames=_labelnames,
            buckets=defs.TOKENS_BUCKETS,
        ),
        "fragmentation": Histogram(
            defs.KV_FRAGMENTATION_RATIO,
            "Per-stage logical KV fragmentation ratio: tail waste tokens / allocated block capacity.",
            labelnames=_labelnames,
            buckets=defs.RATIO_BUCKETS,
        ),
        "cached_tokens": Gauge(
            defs.KV_CACHED_TOKENS,
            "Tokens served from prefix cache in the latest stage snapshot.",
            labelnames=_labelnames,
        ),
        "prefix_hit_ratio": Histogram(
            defs.KV_PREFIX_HIT_RATIO,
            "Per-stage prefix cache hit ratio: cached tokens / prompt tokens.",
            labelnames=_labelnames,
            buckets=defs.RATIO_BUCKETS,
        ),
    }
    return _kv_families


class OmniKVCacheMetrics:
    """Observe API for KV cache efficiency families."""

    def __init__(self, model_name: str, log_stats: bool = True) -> None:
        self._model_name = model_name
        self._log_stats = log_stats

    def observe_snapshot(
        self,
        *,
        stage: int | str,
        replica: int | str | None,
        modality: str | None,
        snapshot: KVEfficiencySnapshot | None,
    ) -> None:
        if not self._log_stats or snapshot is None or replica is None:
            return
        labels = {
            "model_name": self._model_name,
            "stage": str(stage),
            "replica": str(replica),
            "modality": str(modality or "unknown"),
            "source": str(snapshot.source or "unknown"),
        }
        families = _families()
        families["footprint_tokens"].labels(**labels).set(snapshot.footprint_tokens)
        if snapshot.footprint_bytes > 0:
            families["footprint_bytes"].labels(**labels).set(snapshot.footprint_bytes)
        families["block_occupancy"].labels(**labels).observe(snapshot.occupancy_ratio)
        families["tail_waste"].labels(**labels).observe(snapshot.tail_waste_tokens)
        families["fragmentation"].labels(**labels).observe(snapshot.fragmentation_ratio)
        families["cached_tokens"].labels(**labels).set(snapshot.cached_tokens)
        if snapshot.prompt_tokens > 0:
            families["prefix_hit_ratio"].labels(**labels).observe(snapshot.prefix_hit_ratio)


def observe_kv_at_stage_metrics(
    kv_metrics: OmniKVCacheMetrics,
    *,
    stage_id: int,
    replica_id: int | None,
    stage_metrics: Any,
    modality: str | None,
) -> None:
    """Emit a KV snapshot carried by ``StageRequestStats`` if present."""

    snapshot = getattr(stage_metrics, "kv_snapshot", None)
    if snapshot is None:
        return
    kv_metrics.observe_snapshot(
        stage=stage_id,
        replica=replica_id,
        modality=modality,
        snapshot=snapshot,
    )
