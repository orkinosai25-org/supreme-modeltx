from __future__ import annotations

import json
from pathlib import Path

from supreme_modeltx.model_core.eval.benchmark import build_report, evaluate_checkpoints


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_evaluate_checkpoints_scores_code_and_reasoning_tasks():
    tasks = [
        {
            "id": "code",
            "category": "code",
            "prompt": "Write a Python function max_of_two(a, b) that returns the larger integer.",
            "scoring": "keyword_ratio",
            "required_keywords": ["def max_of_two", "return", "if"],
        },
        {
            "id": "reasoning",
            "category": "reasoning",
            "prompt": "All birds have wings. A sparrow is a bird. Does a sparrow have wings? Answer yes or no.",
            "scoring": "contains",
            "expected_answer": "yes",
        },
    ]
    payloads = [
        {
            "checkpoint_path": "/tmp/checkpoint_step_00000010.pt",
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "samples": [
                {
                    "prompt": tasks[0]["prompt"],
                    "completion_text": "def max_of_two(a, b):\n    if a > b:\n        return a\n    return b",
                },
                {
                    "prompt": tasks[1]["prompt"],
                    "completion_text": "Yes, a sparrow has wings.",
                },
            ],
        }
    ]

    results = evaluate_checkpoints(tasks, payloads)
    assert len(results) == 1
    metrics = results[0]["metrics"]
    assert metrics["overall_score"] == 1.0
    assert metrics["code_score"] == 1.0
    assert metrics["reasoning_score"] == 1.0


def test_build_report_reads_samples_and_baselines(tmp_path):
    eval_set = tmp_path / "benchmark_eval_set.json"
    _write_json(
        eval_set,
        {
            "tasks": [
                {
                    "id": "reasoning",
                    "category": "reasoning",
                    "prompt": "If five machines make five widgets in five minutes, how many widgets do 100 machines make in five minutes?",
                    "scoring": "contains",
                    "expected_answer": "100",
                }
            ]
        },
    )

    baselines = tmp_path / "benchmark_baselines.json"
    _write_json(
        baselines,
        {
            "selected_open_baselines": [
                {
                    "name": "Test baseline",
                    "source": "https://example.com",
                    "metrics": {"overall_score": 0.4, "code_score": 0.0, "reasoning_score": 0.8},
                }
            ]
        },
    )

    sample_payload = {
        "checkpoint_path": "/tmp/checkpoint_step_00000020.pt",
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "samples": [
            {
                "prompt": "If five machines make five widgets in five minutes, how many widgets do 100 machines make in five minutes?",
                "completion_text": "100 widgets.",
            }
        ],
    }
    _write_json(tmp_path / "run_artifacts" / "samples" / "checkpoint_step_00000020_samples.json", sample_payload)

    report = build_report(
        eval_set_path=eval_set,
        baselines_path=baselines,
        samples_root=tmp_path / "run_artifacts",
    )
    assert report["benchmark_name"] == "smtx-mini-code-reasoning-v1"
    assert report["best_local_checkpoint"]["metrics"]["overall_score"] == 1.0
    assert report["selected_open_baselines"][0]["name"] == "Test baseline"
