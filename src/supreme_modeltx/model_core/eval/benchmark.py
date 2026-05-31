"""
model_core/eval/benchmark.py — Lightweight benchmark scoring for code/reasoning samples.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _normalise_text(value: str) -> str:
    return " ".join(value.lower().split())


def _score_task(task: dict[str, Any], completion_text: str) -> float:
    scoring = task.get("scoring", "contains")
    haystack = _normalise_text(completion_text)
    if not haystack:
        return 0.0

    if scoring == "keyword_ratio":
        keywords = [_normalise_text(str(k)) for k in task.get("required_keywords", []) if str(k).strip()]
        if not keywords:
            return 0.0
        hits = sum(1 for keyword in keywords if keyword in haystack)
        return hits / len(keywords)

    expected = _normalise_text(str(task.get("expected_answer", "")))
    if not expected:
        return 0.0
    return 1.0 if expected in haystack else 0.0


def _load_tasks(eval_set_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(eval_set_path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", [])
    return [task for task in tasks if isinstance(task, dict)]


def _find_sample_payloads(samples_root: Path) -> list[dict[str, Any]]:
    patterns = [
        "samples/checkpoint_step_*_samples.json",
        "**/samples/checkpoint_step_*_samples.json",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(samples_root.glob(pattern))
    unique_files = sorted(set(path.resolve() for path in files))

    payloads: list[dict[str, Any]] = []
    for path in unique_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_checkpoints(tasks: list[dict[str, Any]], sample_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for payload in sample_payloads:
        rows = payload.get("samples", [])
        prompt_to_completion = {
            str(row.get("prompt", "")): str(row.get("completion_text", ""))
            for row in rows
            if isinstance(row, dict)
        }
        task_rows: list[dict[str, Any]] = []
        code_scores: list[float] = []
        reasoning_scores: list[float] = []

        for task in tasks:
            prompt = str(task.get("prompt", ""))
            completion = prompt_to_completion.get(prompt, "")
            score = _score_task(task, completion)
            category = str(task.get("category", "other"))
            task_rows.append(
                {
                    "id": task.get("id"),
                    "category": category,
                    "prompt": prompt,
                    "score": score,
                    "completion_text": completion,
                    "scoring": task.get("scoring", "contains"),
                }
            )
            if category == "code":
                code_scores.append(score)
            elif category == "reasoning":
                reasoning_scores.append(score)

        overall_scores = [row["score"] for row in task_rows]
        results.append(
            {
                "checkpoint_path": payload.get("checkpoint_path"),
                "generated_at_utc": payload.get("generated_at_utc"),
                "metrics": {
                    "overall_score": round(_mean(overall_scores), 4),
                    "code_score": round(_mean(code_scores), 4),
                    "reasoning_score": round(_mean(reasoning_scores), 4),
                    "task_count": len(task_rows),
                },
                "task_results": task_rows,
            }
        )
    return results


def _load_baselines(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    baselines = payload.get("selected_open_baselines", [])
    return [baseline for baseline in baselines if isinstance(baseline, dict)]


def build_report(
    *,
    eval_set_path: Path,
    baselines_path: Path,
    samples_root: Path,
) -> dict[str, Any]:
    tasks = _load_tasks(eval_set_path)
    sample_payloads = _find_sample_payloads(samples_root)
    checkpoint_results = evaluate_checkpoints(tasks, sample_payloads)
    baselines = _load_baselines(baselines_path)

    best_local = None
    if checkpoint_results:
        best_local = max(checkpoint_results, key=lambda row: row["metrics"]["overall_score"])

    return {
        "benchmark_name": "smtx-mini-code-reasoning-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "eval_set_path": str(eval_set_path),
            "baselines_path": str(baselines_path),
            "samples_root": str(samples_root),
        },
        "methodology": {
            "task_count": len(tasks),
            "local_checkpoint_source": "run_artifacts/samples/checkpoint_step_*_samples.json",
            "scoring": {
                "keyword_ratio": "score = matched required keywords / total required keywords",
                "contains": "score = 1.0 if expected answer appears in completion, else 0.0",
            },
        },
        "limitations": [
            "This benchmark is a compact directional signal, not a replacement for full public benchmark suites.",
            "Scores depend on deterministic checkpoint sample generation prompts and may not measure broad task generalisation.",
            "Open baseline scores are reference values sourced from public model cards or project documentation.",
        ],
        "local_checkpoints": checkpoint_results,
        "best_local_checkpoint": best_local,
        "selected_open_baselines": baselines,
    }


def _build_markdown(report: dict[str, Any]) -> str:
    best = report.get("best_local_checkpoint")
    lines = [
        "# Baseline Benchmark Summary",
        "",
        f"- Benchmark: `{report.get('benchmark_name')}`",
        f"- Generated at (UTC): `{report.get('generated_at_utc')}`",
        f"- Task count: `{report.get('methodology', {}).get('task_count', 0)}`",
        "",
        "## Best local checkpoint",
    ]
    if best:
        metrics = best.get("metrics", {})
        lines.extend(
            [
                f"- Path: `{best.get('checkpoint_path')}`",
                f"- Overall score: `{metrics.get('overall_score')}`",
                f"- Code score: `{metrics.get('code_score')}`",
                f"- Reasoning score: `{metrics.get('reasoning_score')}`",
            ]
        )
    else:
        lines.append("- No local checkpoint samples found.")

    lines.extend(["", "## Selected open baselines"])
    baselines = report.get("selected_open_baselines", [])
    if not baselines:
        lines.append("- None configured.")
    for baseline in baselines:
        metrics = baseline.get("metrics", {})
        lines.extend(
            [
                f"- **{baseline.get('name', 'unknown')}** (`{baseline.get('source', 'n/a')}`): "
                f"overall={metrics.get('overall_score', 'n/a')}, "
                f"code={metrics.get('code_score', 'n/a')}, "
                f"reasoning={metrics.get('reasoning_score', 'n/a')}",
            ]
        )
    lines.extend(["", "## Methodology limitations"])
    for limitation in report.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score SMTX benchmark outputs against mini code/reasoning tasks.")
    parser.add_argument("--eval-set", required=True, help="Path to benchmark eval-set JSON")
    parser.add_argument("--baselines", required=True, help="Path to baseline-reference JSON")
    parser.add_argument("--samples-root", default="run_artifacts", help="Root directory containing samples/")
    parser.add_argument("--output-dir", default="benchmark_outputs", help="Output directory for benchmark artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        eval_set_path=Path(args.eval_set),
        baselines_path=Path(args.baselines),
        samples_root=Path(args.samples_root),
    )

    (output_dir / "benchmark_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "benchmark_results.md").write_text(_build_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
