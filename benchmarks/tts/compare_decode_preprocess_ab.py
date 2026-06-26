#!/usr/bin/env python3
"""Engine-level A/B for the batched decode-preprocess fast path.

Compares two ``vllm bench serve --omni`` runs of the same model/dataset/sweep
with the fast path ON vs OFF, and reports TTFT / E2E / throughput deltas. The
path is selected by a server-side kill-switch, not code changes:

    VLLM_OMNI_DISABLE_BATCH_DECODE_PREPROCESS=1    ->  scalar baseline (A)
    (unset / 0)                                     ->  batched fast path (B)

The switch gates ``OmniGPUModelRunner._resolve_batch_decode_preprocess``, so an
honest A/B needs two server lifetimes (one per setting). This script only parses
the resulting JSON files, so it runs anywhere; it enforces the **no c=1
regression** gate and exits non-zero if that gate fails.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# (json_key, human_label, lower_is_better). Keys match what
# ``vllm bench serve --omni`` writes.
_METRICS: list[tuple[str, str, bool]] = [
    ("mean_ttft_ms", "TTFT mean (ms)", True),
    ("p99_ttft_ms", "TTFT p99 (ms)", True),
    ("mean_e2el_ms", "E2E mean (ms)", True),
    ("p99_e2el_ms", "E2E p99 (ms)", True),
    ("audio_throughput", "Audio throughput", False),
    ("mean_audio_rtf", "RTF mean", True),
    ("mean_audio_ttfp_ms", "TTFP mean (ms)", True),
]

# c=1 regression beyond this fraction fails the gate. 5% absorbs GPU jitter
# while still catching a real single-stream slowdown.
_C1_REGRESSION_TOLERANCE = 0.05


@dataclass
class RunResult:
    """One parsed ``vllm bench serve`` result file, keyed by concurrency."""

    concurrency: int
    task: str
    path: Path
    metrics: dict[str, float] = field(default_factory=dict)


def _coerce_float(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _concurrency_of(payload: dict[str, Any]) -> int:
    """Pull the concurrency level from a result payload.

    ``bench_tts.py`` annotates results with ``_concurrency`` for its own
    summary table; ``vllm bench serve`` also records ``max_concurrency``.
    Fall back across both so we work with either producer.
    """
    for key in ("_concurrency", "max_concurrency", "concurrency"):
        if key in payload and payload[key] is not None:
            try:
                return int(payload[key])
            except (TypeError, ValueError):
                continue
    return -1


def load_results(result_dir: Path) -> dict[int, RunResult]:
    """Load every result_*.json in a directory, keyed by concurrency."""
    if not result_dir.is_dir():
        raise FileNotFoundError(f"result dir not found: {result_dir}")

    out: dict[int, RunResult] = {}
    json_files = sorted(result_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"no .json result files under {result_dir}")

    for jf in json_files:
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[compare] WARNING: skipping unreadable {jf.name}: {exc}", file=sys.stderr)
            continue
        if not isinstance(payload, dict):
            continue
        conc = _concurrency_of(payload)
        if conc < 0:
            print(f"[compare] WARNING: {jf.name} has no concurrency field; skipping", file=sys.stderr)
            continue
        metrics = {key: _coerce_float(payload.get(key)) for key, _, _ in _METRICS}
        # Later files for the same concurrency win (most recent run).
        out[conc] = RunResult(
            concurrency=conc,
            task=str(payload.get("_task", payload.get("dataset_name", "?"))),
            path=jf,
            metrics=metrics,
        )
    return out


def _pct_delta(scalar: float, batched: float, lower_is_better: bool) -> float:
    """Signed improvement percentage of batched relative to scalar.

    Positive = batched is better. For lower-is-better metrics that means a
    reduction; for higher-is-better it means an increase.
    """
    if math.isnan(scalar) or math.isnan(batched) or scalar == 0:
        return float("nan")
    raw = (batched - scalar) / abs(scalar) * 100.0
    return -raw if lower_is_better else raw


@dataclass
class GateResult:
    passed: bool
    messages: list[str] = field(default_factory=list)


def check_c1_no_regression(
    scalar: dict[int, RunResult],
    batched: dict[int, RunResult],
    tolerance: float = _C1_REGRESSION_TOLERANCE,
) -> GateResult:
    """Reviewer's hard gate: batched must not regress the c=1 case.

    We only police latency (lower-is-better) metrics at concurrency 1, since
    the whole point of the fast path is to help at high concurrency *without*
    taxing the single-stream path.
    """
    result = GateResult(passed=True)
    if 1 not in scalar or 1 not in batched:
        result.messages.append("c=1 run missing in one side; cannot evaluate no-regression gate")
        return result

    a, b = scalar[1].metrics, batched[1].metrics
    for key, label, lower_is_better in _METRICS:
        if not lower_is_better:
            continue
        sa, sb = a.get(key, float("nan")), b.get(key, float("nan"))
        if math.isnan(sa) or math.isnan(sb) or sa == 0:
            continue
        regression = (sb - sa) / abs(sa)
        if regression > tolerance:
            result.passed = False
            result.messages.append(
                f"c=1 REGRESSION on {label}: scalar={sa:.3f} -> batched={sb:.3f} "
                f"(+{regression * 100:.1f}% > {tolerance * 100:.0f}% tol)"
            )
    if result.passed and not result.messages:
        result.messages.append("c=1 no-regression gate: PASS")
    return result


def format_table(scalar: dict[int, RunResult], batched: dict[int, RunResult]) -> str:
    concurrencies = sorted(set(scalar) & set(batched))
    if not concurrencies:
        return "(no overlapping concurrency levels between scalar and batched)"

    lines: list[str] = []
    for conc in concurrencies:
        a, b = scalar[conc].metrics, batched[conc].metrics
        lines.append(f"\n## concurrency = {conc}")
        header = f"{'metric':<20} {'scalar (A)':>14} {'batched (B)':>14} {'improvement':>13}"
        lines.append(header)
        lines.append("-" * len(header))
        for key, label, lower_is_better in _METRICS:
            sa, sb = a.get(key, float("nan")), b.get(key, float("nan"))
            delta = _pct_delta(sa, sb, lower_is_better)

            def fmt(v: float) -> str:
                return f"{v:.3f}" if not math.isnan(v) else "n/a"

            delta_str = f"{delta:+.1f}%" if not math.isnan(delta) else "n/a"
            lines.append(f"{label:<20} {fmt(sa):>14} {fmt(sb):>14} {delta_str:>13}")
    return "\n".join(lines)


def build_payload(scalar: dict[int, RunResult], batched: dict[int, RunResult], gate: GateResult) -> dict[str, Any]:
    concurrencies = sorted(set(scalar) & set(batched))
    per_conc: list[dict[str, Any]] = []
    for conc in concurrencies:
        a, b = scalar[conc].metrics, batched[conc].metrics
        metric_rows = {}
        for key, label, lower_is_better in _METRICS:
            metric_rows[key] = {
                "label": label,
                "scalar": a.get(key, float("nan")),
                "batched": b.get(key, float("nan")),
                "improvement_pct": _pct_delta(a.get(key, float("nan")), b.get(key, float("nan")), lower_is_better),
            }
        per_conc.append({"concurrency": conc, "metrics": metric_rows})
    return {
        "benchmark": "decode_preprocess_ab",
        "description": "Engine-level A/B of batched decode-preprocess: scalar baseline vs batched.",
        "c1_no_regression_passed": gate.passed,
        "gate_messages": gate.messages,
        "results": per_conc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scalar", type=Path, required=True, help="Result dir from the scalar-baseline (A) run")
    parser.add_argument("--batched", type=Path, required=True, help="Result dir from the batched (B) run")
    parser.add_argument("--output-json", type=Path, default=None, help="Write the structured comparison here")
    parser.add_argument(
        "--c1-tolerance",
        type=float,
        default=_C1_REGRESSION_TOLERANCE,
        help="Allowed c=1 latency regression fraction before the gate fails (default 0.05)",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Report the c=1 gate but do not exit non-zero on failure (analysis-only mode)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scalar = load_results(args.scalar)
    batched = load_results(args.batched)

    print(format_table(scalar, batched))

    gate = check_c1_no_regression(scalar, batched, tolerance=args.c1_tolerance)
    print("\n" + "=" * 60)
    for msg in gate.messages:
        print(msg)
    print("=" * 60)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(build_payload(scalar, batched, gate), indent=2), encoding="utf-8")
        print(f"\nWrote comparison JSON to {args.output_json}")

    if not gate.passed and not args.no_gate:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
