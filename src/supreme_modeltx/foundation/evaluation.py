from __future__ import annotations

import argparse
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

    results: list[dict[str, Any]] = []
    for case in suite.cases:
        started = time.perf_counter()
        response = responder.generate(case.prompt)
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

    summary = {
        "suite_name": suite.suite_name,
        "suite_version": suite.version,
        "total_cases": len(results),
        "passed_cases": sum(1 for result in results if result["passed"]),
        "results": results,
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
