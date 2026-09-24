import json

from supreme_modeltx.foundation.config import EvaluationRunConfig
from supreme_modeltx.foundation.evaluation import run_evaluation


def test_evaluation_writes_json_results(tmp_path):
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "\n".join(
            [
                "suite_name: smoke-suite",
                "version: v1",
                "cases:",
                "  - case_id: case-1",
                '    prompt: "hello"',
                "    tags: [smoke]",
                "    expected:",
                "      required_substrings:",
                '        - "Draft pilot response:"',
            ]
        ),
        encoding="utf-8",
    )

    config = EvaluationRunConfig.model_validate(
        {
            "evaluation": {
                "suite_path": str(suite_path),
                "output_path": str(tmp_path / "results.json"),
            },
            "inference": {
                "backend": "stub",
                "device": "cpu",
                "response_prefix": "Draft pilot response:",
            },
        }
    )

    summary = run_evaluation(config)

    assert summary["passed_cases"] == 1
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert payload["suite_version"] == "v1"
    assert payload["results"][0]["passed"] is True
