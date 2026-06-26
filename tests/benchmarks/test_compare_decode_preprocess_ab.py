from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks" / "tts"))
import compare_decode_preprocess_ab as ab

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _write_result(dir_path: Path, *, concurrency: int, **metrics: float) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    payload = {"_concurrency": concurrency, "_task": "voice_clone", **metrics}
    (dir_path / f"result_c{concurrency}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_results_keys_by_concurrency(tmp_path):
    d = tmp_path / "scalar"
    _write_result(d, concurrency=1, mean_ttft_ms=100.0)
    _write_result(d, concurrency=8, mean_ttft_ms=200.0)

    loaded = ab.load_results(d)

    assert set(loaded) == {1, 8}
    assert loaded[1].metrics["mean_ttft_ms"] == pytest.approx(100.0)
    assert loaded[8].metrics["mean_ttft_ms"] == pytest.approx(200.0)


def test_load_results_falls_back_to_max_concurrency_field(tmp_path):
    d = tmp_path / "scalar"
    d.mkdir(parents=True)
    # No _concurrency key, only vllm-native max_concurrency.
    (d / "result.json").write_text(json.dumps({"max_concurrency": 16, "mean_ttft_ms": 50.0}), encoding="utf-8")

    loaded = ab.load_results(d)

    assert set(loaded) == {16}


def test_pct_delta_lower_is_better_positive_when_batched_faster():
    # scalar 200ms -> batched 100ms on a latency metric is a 50% improvement.
    assert ab._pct_delta(200.0, 100.0, lower_is_better=True) == pytest.approx(50.0)


def test_pct_delta_higher_is_better_positive_when_batched_higher():
    # scalar 10 -> batched 15 on throughput is a +50% improvement.
    assert ab._pct_delta(10.0, 15.0, lower_is_better=False) == pytest.approx(50.0)


def test_c1_gate_passes_when_batched_not_slower(tmp_path):
    scalar_dir, batched_dir = tmp_path / "a", tmp_path / "b"
    _write_result(scalar_dir, concurrency=1, mean_ttft_ms=100.0, mean_e2el_ms=500.0)
    _write_result(batched_dir, concurrency=1, mean_ttft_ms=98.0, mean_e2el_ms=505.0)

    gate = ab.check_c1_no_regression(ab.load_results(scalar_dir), ab.load_results(batched_dir))

    assert gate.passed is True


def test_c1_gate_fails_on_regression_beyond_tolerance(tmp_path):
    scalar_dir, batched_dir = tmp_path / "a", tmp_path / "b"
    _write_result(scalar_dir, concurrency=1, mean_ttft_ms=100.0)
    # +20% TTFT at c=1 is a real single-stream regression -> must fail.
    _write_result(batched_dir, concurrency=1, mean_ttft_ms=120.0)

    gate = ab.check_c1_no_regression(ab.load_results(scalar_dir), ab.load_results(batched_dir))

    assert gate.passed is False
    assert any("REGRESSION" in m for m in gate.messages)


def test_c1_gate_tolerates_small_jitter(tmp_path):
    scalar_dir, batched_dir = tmp_path / "a", tmp_path / "b"
    _write_result(scalar_dir, concurrency=1, mean_ttft_ms=100.0)
    # +3% is within the 5% jitter tolerance.
    _write_result(batched_dir, concurrency=1, mean_ttft_ms=103.0)

    gate = ab.check_c1_no_regression(ab.load_results(scalar_dir), ab.load_results(batched_dir))

    assert gate.passed is True


def test_build_payload_records_gate_and_per_concurrency(tmp_path):
    scalar_dir, batched_dir = tmp_path / "a", tmp_path / "b"
    _write_result(scalar_dir, concurrency=1, mean_ttft_ms=100.0, audio_throughput=10.0)
    _write_result(batched_dir, concurrency=1, mean_ttft_ms=90.0, audio_throughput=12.0)

    scalar, batched = ab.load_results(scalar_dir), ab.load_results(batched_dir)
    gate = ab.check_c1_no_regression(scalar, batched)
    payload = ab.build_payload(scalar, batched, gate)

    assert payload["c1_no_regression_passed"] is True
    assert payload["results"][0]["concurrency"] == 1
    ttft_row = payload["results"][0]["metrics"]["mean_ttft_ms"]
    assert ttft_row["improvement_pct"] == pytest.approx(10.0)
