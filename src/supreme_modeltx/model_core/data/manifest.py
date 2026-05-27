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
    weight: float = Field(1.0, gt=0.0, description="Sampling weight relative to other sources.")
    text_field: str = Field("text", description="JSON key containing the text, for jsonl backends.")


class DataManifest(BaseModel):
    """Root manifest object."""

    version: str = "1"
    description: str = ""
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
