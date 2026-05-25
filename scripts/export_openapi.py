#!/usr/bin/env python3
"""
export_openapi.py — Export the SUMOTX Platform API OpenAPI spec to docs/

Usage:
    python scripts/export_openapi.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from api.main import app

DOCS_DIR = Path(__file__).parent.parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)

schema = app.openapi()

# ── YAML ──────────────────────────────────────────────────────────────────────
yaml_path = DOCS_DIR / "openapi.yaml"
with yaml_path.open("w", encoding="utf-8") as f:
    yaml.dump(schema, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
print(f"Written {yaml_path}")

# ── JSON ──────────────────────────────────────────────────────────────────────
json_path = DOCS_DIR / "openapi.json"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2, default=str)
print(f"Written {json_path}")
