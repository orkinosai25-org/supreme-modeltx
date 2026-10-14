from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from supreme_modeltx.foundation.config import (
    EvaluationRunConfig,
    InferenceRunConfig,
    PromptSuiteConfig,
    load_model,
)
from supreme_modeltx.foundation.inference import FoundationResponder


def evaluate_case(response: str, expected: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    exact_match = expected.get("exact_match")
    if exact_match is not None and response != exact_match:
        failures.append("exact_match")
    for snippet in expected.get("required_substrings", []):
        if snippet not in response:
            failures.append(f"missing:{snippet}")
    for snippet in expected.get("forbidden_substrings", []):
        if snippet in response:
            failures.append(f"forbidden:{snippet}")
    max_characters = expected.get("max_characters")
    if max_characters is not None and len(response) > max_characters:
        failures.append("max_characters")
    return failures


def run_evaluation(config: EvaluationRunConfig) -> dict[str, Any]:
    suite = load_model(config.evaluation.suite_path, PromptSuiteConfig)
    responder = FoundationResponder(InferenceRunConfig(model=config.model, inference=config.inference))
    cuda_measurement = responder.provider is not None and responder.provider.device == "cuda"
    if cuda_measurement:
        import torch

        torch.cuda.reset_peak_memory_stats()

    results: list[dict[str, Any]] = []
    for case in suite.cases:
        if cuda_measurement:
            torch.cuda.synchronize()
        started = time.perf_counter()
        response = responder.generate(case.prompt)
        if cuda_measurement:
            torch.cuda.synchronize()
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        failures = evaluate_case(response, case.expected.model_dump(mode="json"))
        results.append(
            {
                "case_id": case.case_id,
                "passed": not failures,
                "prompt": case.prompt,
                "response": response,
                "latency_ms": duration_ms,
                "tags": case.tags,
                "failures": failures,
            }
        )
        if responder.last_usage is not None:
            results[-1]["prompt_tokens"], results[-1]["completion_tokens"] = responder.last_usage

    summary = {
        "suite_name": suite.suite_name,
        "suite_version": suite.version,
        "total_cases": len(results),
        "passed_cases": sum(1 for result in results if result["passed"]),
        "results": results,
    }
    if responder.provider is not None:
        duration_seconds = sum(result["latency_ms"] for result in results) / 1000
        completion_tokens = sum(result["completion_tokens"] for result in results)
        rate = config.evaluation.compute_cost_per_hour_usd
        summary["runtime"] = {
            "provider": responder.provider.metadata,
            "config": config.model_dump(mode="json"),
            "suite_sha256": hashlib.sha256(Path(config.evaluation.suite_path).read_bytes()).hexdigest(),
            "generation_seconds": duration_seconds,
            "completion_tokens": completion_tokens,
            "completion_tokens_per_second": completion_tokens / duration_seconds if duration_seconds else None,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if cuda_measurement else None,
            "compute_cost_per_hour_usd": rate,
            "estimated_generation_cost_usd": rate * duration_seconds / 3600 if rate is not None else None,
            "measurement_scope": "Serial prompt generation including tokenization and decoding, without a warmup stage; excludes model loading and idle time. Not a serving SLO or total run cost.",
        }
    output_path = Path(config.evaluation.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supreme Model T-X evaluation harness.")
    parser.add_argument(
        "--config",
        default="configs/foundation/evaluation.yaml",
        help="Path to an evaluation config file.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = EvaluationRunConfig.from_file(args.config)
    print(json.dumps(run_evaluation(config), indent=2))


if __name__ == "__main__":
    main()
