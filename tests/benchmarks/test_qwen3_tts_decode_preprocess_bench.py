from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks" / "tts"))
import bench_qwen3_tts_decode_preprocess as bench

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_summarize_times_reports_speedup_and_per_request_costs():
    result = bench.summarize_times(
        batch_size=4,
        scalar_times_ms=[4.0, 6.0, 5.0],
        batched_times_ms=[1.0, 2.0, 1.5],
    )

    assert result.batch_size == 4
    assert result.scalar.mean_ms == pytest.approx(5.0)
    assert result.batched.mean_ms == pytest.approx(1.5)
    assert result.speedup == pytest.approx(5.0 / 1.5)
    assert result.scalar_us_per_request == pytest.approx(1250.0)
    assert result.batched_us_per_request == pytest.approx(375.0)
    assert result.saved_pct == pytest.approx(70.0)


def test_format_results_table_includes_key_columns():
    result = bench.summarize_times(
        batch_size=2,
        scalar_times_ms=[2.0],
        batched_times_ms=[1.0],
    )

    table = bench.format_results_table([result])

    assert "scalar mean ms" in table
    assert "batch mean ms" in table
    assert "speedup" in table
    assert "      2 " in table
