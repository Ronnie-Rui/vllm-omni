from __future__ import annotations

from types import SimpleNamespace

import pytest
from prometheus_client import REGISTRY, generate_latest

from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics import kv as kv_mod
from vllm_omni.metrics.kv import (
    OmniKVCacheMetrics,
    compute_kv_efficiency,
    estimate_kv_bytes_per_token,
    resolve_kv_block_size,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_MODEL = "test-kv-model"


def _sample_value(output: str, line_prefix: str) -> float | None:
    for line in output.splitlines():
        if line.startswith(line_prefix):
            return float(line.split()[-1])
    return None


class TestComputeKVEfficiency:
    def test_full_blocks_have_no_tail_waste(self) -> None:
        snap = compute_kv_efficiency(
            block_size=16,
            sequence_tokens=32,
            bytes_per_token=128,
        )
        assert snap.allocated_blocks == 2
        assert snap.allocated_tokens == 32
        assert snap.tail_waste_tokens == 0
        assert snap.occupancy_ratio == 1.0
        assert snap.fragmentation_ratio == 0.0
        assert snap.footprint_tokens == 32
        assert snap.footprint_bytes == 4096

    def test_partial_tail_waste_is_capacity_minus_useful_tokens(self) -> None:
        snap = compute_kv_efficiency(block_size=16, sequence_tokens=33)
        assert snap.allocated_blocks == 3
        assert snap.allocated_tokens == 48
        assert snap.tail_waste_tokens == 15
        assert snap.occupancy_ratio == pytest.approx(33 / 48)
        assert snap.fragmentation_ratio == pytest.approx(15 / 48)

    def test_prefix_hit_ratio_uses_prompt_tokens(self) -> None:
        snap = compute_kv_efficiency(
            block_size=16,
            sequence_tokens=33,
            cached_tokens=11,
            prompt_tokens=44,
        )
        assert snap.cached_tokens == 11
        assert snap.prompt_tokens == 44
        assert snap.prefix_hit_ratio == pytest.approx(0.25)

    def test_explicit_overallocation_counts_as_fragmentation(self) -> None:
        snap = compute_kv_efficiency(
            block_size=16,
            sequence_tokens=17,
            allocated_blocks=4,
        )
        assert snap.allocated_tokens == 64
        assert snap.tail_waste_tokens == 47
        assert snap.occupancy_ratio == pytest.approx(17 / 64)
        assert snap.fragmentation_ratio == pytest.approx(47 / 64)

    def test_empty_snapshot_is_zero_safe(self) -> None:
        snap = compute_kv_efficiency(block_size=16, sequence_tokens=0)
        assert snap.allocated_blocks == 0
        assert snap.allocated_tokens == 0
        assert snap.occupancy_ratio == 0.0
        assert snap.fragmentation_ratio == 0.0


class _ModelConfig:
    dtype = "float16"

    def get_num_layers(self, parallel_config=None):
        return 24

    def get_num_kv_heads(self, parallel_config=None):
        return 8

    def get_head_size(self):
        return 64


class TestConfigResolution:
    def test_resolve_block_size_from_cache_config(self) -> None:
        cfg = SimpleNamespace(cache_config=SimpleNamespace(block_size=32))
        assert resolve_kv_block_size(cfg) == 32

    def test_estimate_bytes_per_token_from_vllm_config(self) -> None:
        cfg = SimpleNamespace(
            model_config=_ModelConfig(),
            cache_config=SimpleNamespace(cache_dtype="auto"),
            parallel_config=SimpleNamespace(),
        )
        assert estimate_kv_bytes_per_token(cfg) == 24 * 2 * 8 * 64 * 2


class TestPrometheusObserve:
    def test_all_families_present_with_expected_labels(self) -> None:
        metrics = OmniKVCacheMetrics(model_name=_MODEL)
        snap = compute_kv_efficiency(
            block_size=16,
            sequence_tokens=33,
            bytes_per_token=128,
            cached_tokens=3,
            prompt_tokens=12,
        )
        metrics.observe_snapshot(
            stage=1,
            replica=0,
            modality="text",
            snapshot=snap,
        )
        out = generate_latest(REGISTRY).decode()
        for family in (
            defs.KV_FOOTPRINT_TOKENS,
            defs.KV_FOOTPRINT_BYTES,
            defs.KV_BLOCK_OCCUPANCY_RATIO,
            defs.KV_TAIL_WASTE_TOKENS,
            defs.KV_FRAGMENTATION_RATIO,
            defs.KV_PREFIX_HIT_RATIO,
        ):
            assert f"# HELP {family}" in out
        assert f"# HELP {defs.KV_CACHED_TOKENS}" in out

        labels = 'modality="text",model_name="test-kv-model",replica="0",source="stage_tokens",stage="1"'
        assert (
            _sample_value(
                out,
                f"{defs.KV_FOOTPRINT_TOKENS}{{{labels}}}",
            )
            == 48.0
        )
        assert (
            _sample_value(
                out,
                f"{defs.KV_FOOTPRINT_BYTES}{{{labels}}}",
            )
            == 6144.0
        )
        assert (
            _sample_value(
                out,
                f"{defs.KV_TAIL_WASTE_TOKENS}_sum{{{labels}}}",
            )
            == 15.0
        )
        assert (
            _sample_value(
                out,
                f"{defs.KV_CACHED_TOKENS}{{{labels}}}",
            )
            == 3.0
        )

    def test_cached_tokens_gauge_uses_latest_snapshot(self) -> None:
        metrics = OmniKVCacheMetrics(model_name=_MODEL)
        first = compute_kv_efficiency(
            block_size=16,
            sequence_tokens=16,
            cached_tokens=5,
            prompt_tokens=10,
        )
        second = compute_kv_efficiency(
            block_size=16,
            sequence_tokens=16,
            cached_tokens=2,
            prompt_tokens=10,
        )
        metrics.observe_snapshot(stage=2, replica=0, modality="text", snapshot=first)
        metrics.observe_snapshot(stage=2, replica=0, modality="text", snapshot=second)

        out = generate_latest(REGISTRY).decode()
        labels = 'modality="text",model_name="test-kv-model",replica="0",source="stage_tokens",stage="2"'
        assert (
            _sample_value(
                out,
                f"{defs.KV_CACHED_TOKENS}{{{labels}}}",
            )
            == 2.0
        )

    def test_families_recreated_after_registry_unregister(self) -> None:
        metrics = OmniKVCacheMetrics(model_name=_MODEL)
        snap = compute_kv_efficiency(block_size=16, sequence_tokens=16)
        metrics.observe_snapshot(stage=3, replica=0, modality="text", snapshot=snap)

        assert kv_mod._kv_families is not None
        for family in kv_mod._kv_families.values():
            REGISTRY.unregister(family)

        metrics.observe_snapshot(stage=3, replica=0, modality="text", snapshot=snap)

        out = generate_latest(REGISTRY).decode()
        labels = 'modality="text",model_name="test-kv-model",replica="0",source="stage_tokens",stage="3"'
        assert _sample_value(out, f"{defs.KV_FOOTPRINT_TOKENS}{{{labels}}}") == 16.0


class TestStagePoolSnapshot:
    def test_llm_stage_metrics_carries_kv_snapshot_without_changing_prompt_counter(self) -> None:
        from vllm_omni.engine.stage_pool import StagePool, _ReplicaMetrics

        pool = object.__new__(StagePool)
        pool.stage_id = 1
        pool.clients = [
            SimpleNamespace(
                stage_type="llm",
                final_output_type="text",
            )
        ]
        pool._output_processor = SimpleNamespace(pop_native_text_metrics=lambda request_id: {})
        pool._stage_vllm_config = None
        pool._kv_block_size = 16
        pool._kv_bytes_per_token = 128
        pool._replica_metrics = [_ReplicaMetrics()]
        pool._output_timestamps_by_request = {}
        pool._non_empty_first_output_timestamps_by_request = {}
        pool._audio_frames_by_request = {}
        pool._audio_sample_rate_by_request = {}

        request_output = SimpleNamespace(
            request_id="req-kv",
            prompt_token_ids=list(range(21)),
            outputs=[SimpleNamespace(token_ids=list(range(5)), cumulative_token_ids=None)],
            final_output_type="text",
            images=[],
            video=None,
            videos=None,
            trajectory_latents=None,
        )

        stats = pool.build_stage_metrics(
            [request_output],
            submit_ts=10.0,
            request_timestamp=10.0,
            replica_id=0,
        )

        assert stats.num_tokens_in == 0
        assert stats.num_tokens_out == 5
        assert stats.kv_snapshot is not None
        assert stats.kv_snapshot.sequence_tokens == 26
        assert stats.kv_snapshot.allocated_tokens == 32
        assert stats.kv_snapshot.tail_waste_tokens == 6
        assert stats.kv_snapshot.footprint_bytes == 4096

    def test_stage_metrics_prefers_real_block_ids_when_present(self) -> None:
        from vllm_omni.engine.stage_pool import StagePool, _ReplicaMetrics

        pool = object.__new__(StagePool)
        pool.stage_id = 0
        pool.clients = [
            SimpleNamespace(
                stage_type="llm",
                final_output_type="text",
            )
        ]
        pool._output_processor = SimpleNamespace(pop_native_text_metrics=lambda request_id: {})
        pool._stage_vllm_config = None
        pool._kv_block_size = 16
        pool._kv_bytes_per_token = 128
        pool._replica_metrics = [_ReplicaMetrics()]
        pool._output_timestamps_by_request = {}
        pool._non_empty_first_output_timestamps_by_request = {}
        pool._audio_frames_by_request = {}
        pool._audio_sample_rate_by_request = {}

        request_output = SimpleNamespace(
            request_id="req-kv-real-blocks",
            prompt_token_ids=list(range(17)),
            outputs=[SimpleNamespace(token_ids=[], cumulative_token_ids=None)],
            final_output_type="text",
            images=[],
            video=None,
            videos=None,
            trajectory_latents=None,
            kv_transfer_params={"metadata": {"block_ids": [11, 12, 13, 14]}},
        )

        stats = pool.build_stage_metrics(
            [request_output],
            submit_ts=10.0,
            request_timestamp=10.0,
            replica_id=0,
        )

        assert stats.kv_snapshot is not None
        assert stats.kv_snapshot.source == "kv_transfer_params"
        assert stats.kv_snapshot.allocated_blocks == 4
        assert stats.kv_snapshot.allocated_tokens == 64
        assert stats.kv_snapshot.tail_waste_tokens == 47

    def test_stage_metrics_prefers_scheduler_kv_cache_payload(self) -> None:
        from vllm_omni.engine.stage_pool import StagePool, _ReplicaMetrics

        pool = object.__new__(StagePool)
        pool.stage_id = 0
        pool.clients = [
            SimpleNamespace(
                stage_type="llm",
                final_output_type="text",
            )
        ]
        pool._output_processor = SimpleNamespace(pop_native_text_metrics=lambda request_id: {})
        pool._stage_vllm_config = None
        pool._kv_block_size = 16
        pool._kv_bytes_per_token = 128
        pool._replica_metrics = [_ReplicaMetrics()]
        pool._output_timestamps_by_request = {}
        pool._non_empty_first_output_timestamps_by_request = {}
        pool._audio_frames_by_request = {}
        pool._audio_sample_rate_by_request = {}

        request_output = SimpleNamespace(
            request_id="req-kv-cache-manager",
            prompt_token_ids=list(range(11)),
            outputs=[SimpleNamespace(token_ids=list(range(4)), cumulative_token_ids=None)],
            final_output_type="text",
            images=[],
            video=None,
            videos=None,
            trajectory_latents=None,
            kv_transfer_params={"metadata": {"block_ids": [99]}},
        )
        request_output._omni_kv_cache_metrics = {
            "seq_len": 35,
            "block_ids": [11, 12, 13],
            "source": "kv_cache_manager",
        }

        stats = pool.build_stage_metrics(
            [request_output],
            submit_ts=10.0,
            request_timestamp=10.0,
            replica_id=0,
        )

        assert stats.kv_snapshot is not None
        assert stats.kv_snapshot.source == "kv_cache_manager"
        assert stats.kv_snapshot.sequence_tokens == 35
        assert stats.kv_snapshot.allocated_blocks == 3
        assert stats.kv_snapshot.allocated_tokens == 48
        assert stats.kv_snapshot.tail_waste_tokens == 13


class TestSchedulerKVPayload:
    def test_scheduler_payload_reads_allocator_block_ids(self) -> None:
        from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

        class _KVCacheManager:
            def get_block_ids(self, request_id):
                assert request_id == "req-kv"
                return ([10, 11, 12],)

        scheduler = object.__new__(OmniSchedulerMixin)
        scheduler.kv_cache_manager = _KVCacheManager()
        request = SimpleNamespace(
            request_id="req-kv",
            num_computed_tokens=35,
            prompt_token_ids=list(range(11)),
            output_token_ids=list(range(24)),
        )

        payload = scheduler._build_kv_cache_metrics_payload(request)

        assert payload == {
            "seq_len": 35,
            "block_ids": [10, 11, 12],
            "source": "kv_cache_manager",
        }
