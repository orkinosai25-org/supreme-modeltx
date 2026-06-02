"""
model_core/data/manifest.py — Data manifest schema and loader.

A manifest is a YAML or JSON file that declares one or more data sources
for a training or evaluation run.  Each source specifies a backend
(jsonl, parquet, hf_dataset, text) and its path or identifier.

Example manifest (YAML):
    version: "1"
    sources:
      - name: wiki_style
        backend: text
        path: data/raw/wiki_style
      - name: conversations
        backend: jsonl
        path: data/raw/conversations/conversations.jsonl
      - name: instructions
        backend: jsonl
        path: data/raw/instructions/instructions.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataSource(BaseModel):
    """A single data source declaration inside a manifest."""

    name: str = Field(..., description="Human-readable identifier.")
    backend: Literal["jsonl", "parquet", "hf_dataset", "text"] = "jsonl"
    path: str | None = None
    hf_name: str | None = None
    hf_split: str = "train"
    split: str = Field("train", description="Logical split label (e.g. train, validation, test).")
    weight: float = Field(1.0, gt=0.0, description="Sampling weight relative to other sources.")
    text_field: str = Field("text", description="JSON key containing the text, for jsonl backends.")
    category: Literal["code", "reasoning", "documentation", "synthetic"] | None = None
    provenance: str | None = Field(
        None,
        description="Human-readable source reference for licensing and auditability.",
    )
    license: str | None = Field(None, description="License or usage basis for this source.")
    purpose: str | None = Field(None, description="Why this source exists in the corpus mix.")
    inclusion_rationale: str | None = Field(
        None,
        description="Why the source is included despite other available options.",
    )
    preprocessing: list[str] = Field(
        default_factory=list,
        description="Source-specific preparation expectations before training.",
    )
    benchmark_alignment: list[str] = Field(
        default_factory=list,
        description="Benchmark tasks or prompt families this source is intended to support.",
    )


class DataManifest(BaseModel):
    """Root manifest object."""

    version: str = "1"
    description: str = ""
    name: str = ""
    manifest_type: Literal["training_run", "corpus_plan"] = "training_run"
    status: Literal["materialized", "planned"] = "materialized"
    corpus_version: str | None = None
    previous_baseline: str | None = None
    split_rules: dict[str, str] = Field(default_factory=dict)
    preprocessing_expectations: list[str] = Field(default_factory=list)
    benchmark_alignment: list[str] = Field(default_factory=list)
    changes_from_previous: list[str] = Field(default_factory=list)
    sources: list[DataSource] = Field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "DataManifest":
        path = Path(path)
        raw = path.read_text()
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False)

    def resolve_source_path(self, source: DataSource, *, base_dir: str | Path | None = None) -> Path | None:
        """Resolve a source path against an optional base directory."""
        if not source.path:
            return None
        path = Path(source.path)
        if path.is_absolute() or base_dir is None:
            return path
        return Path(base_dir) / path

    def validate_manifest(self, *, base_dir: str | Path | None = None) -> list[str]:
        """Return lightweight structural validation errors for a manifest."""
        errors: list[str] = []
        seen_names: set[str] = set()
        defined_splits = set(self.split_rules)
        observed_splits = {source.split for source in self.sources}

        if not self.sources:
            errors.append("Manifest must define at least one source.")

        if self.manifest_type == "corpus_plan":
            if not self.name:
                errors.append("Corpus-plan manifest must define a name.")
            if not self.corpus_version:
                errors.append("Corpus-plan manifest must define corpus_version.")
            if not self.previous_baseline:
                errors.append("Corpus-plan manifest must define previous_baseline.")
            if not self.preprocessing_expectations:
                errors.append("Corpus-plan manifest must define preprocessing_expectations.")
            if not self.benchmark_alignment:
                errors.append("Corpus-plan manifest must define benchmark_alignment.")
            if not self.changes_from_previous:
                errors.append("Corpus-plan manifest must define changes_from_previous.")
            if not {"train", "validation"}.issubset(defined_splits):
                errors.append("Corpus-plan manifest must define train and validation split_rules.")
            if not {"train", "validation"}.issubset(observed_splits):
                errors.append("Corpus-plan manifest must include train and validation sources.")

        for source in self.sources:
            if source.name in seen_names:
                errors.append(f"Duplicate manifest source name: {source.name}")
            seen_names.add(source.name)

            if defined_splits and source.split not in defined_splits:
                errors.append(
                    f"Source '{source.name}' uses split '{source.split}' not declared in split_rules."
                )

            if source.backend == "hf_dataset":
                if not source.hf_name:
                    errors.append(f"HF dataset source '{source.name}' must define hf_name.")
            elif not source.path:
                errors.append(f"Manifest source '{source.name}' has no path configured.")
            elif self.status == "materialized":
                resolved = self.resolve_source_path(source, base_dir=base_dir)
                if resolved is not None and not resolved.exists():
                    errors.append(
                        f"Materialized manifest source '{source.name}' path does not exist: {resolved}"
                    )

            if self.manifest_type != "corpus_plan":
                continue

            if not source.category:
                errors.append(f"Corpus-plan source '{source.name}' must define category.")
            if not source.provenance:
                errors.append(f"Corpus-plan source '{source.name}' must define provenance.")
            if not source.license:
                errors.append(f"Corpus-plan source '{source.name}' must define license.")
            if not source.purpose:
                errors.append(f"Corpus-plan source '{source.name}' must define purpose.")
            if not source.inclusion_rationale:
                errors.append(
                    f"Corpus-plan source '{source.name}' must define inclusion_rationale."
                )
            if not source.preprocessing:
                errors.append(f"Corpus-plan source '{source.name}' must define preprocessing.")
            if not source.benchmark_alignment:
                errors.append(
                    f"Corpus-plan source '{source.name}' must define benchmark_alignment."
                )

        return errors
