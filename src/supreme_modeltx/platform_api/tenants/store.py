"""platform_api/tenants/store.py — SQLite-backed project store."""
from __future__ import annotations

import sqlite3
from typing import Optional

from supreme_modeltx.platform_api.persistence.sqlite import connect, resolve_db_path
from supreme_modeltx.platform_api.tenants.models import Project, ProjectCreate


class ProjectStore:
    """SQLite-backed project registry."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = resolve_db_path(db_path)
        self._initialize()
        self._seed_dev_project()

    def list_projects(self) -> list[Project]:
        with connect(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, name, description, owner_email, created_at, is_active
                FROM projects
                ORDER BY created_at ASC
                """
            ).fetchall()
        return [self._row_to_project(row) for row in rows]

    def get_project(self, project_id: str) -> Optional[Project]:
        with connect(self._db_path) as conn:
            row = conn.execute(
                """
                SELECT id, name, description, owner_email, created_at, is_active
                FROM projects
                WHERE id = ?
                """,
                (project_id,),
            ).fetchone()
        return self._row_to_project(row) if row else None

    def create_project(self, body: ProjectCreate) -> Project:
        project = Project(name=body.name, description=body.description, owner_email=body.owner_email)
        with connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO projects (id, name, description, owner_email, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    project.id,
                    project.name,
                    project.description,
                    project.owner_email,
                    project.created_at.isoformat(),
                    int(project.is_active),
                ),
            )
            conn.commit()
        return project

    def _initialize(self) -> None:
        with connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    owner_email TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.commit()

    def _seed_dev_project(self) -> None:
        dev = Project(id="dev-project", name="Dev Project", description="Default development project")
        with connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO projects (id, name, description, owner_email, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    dev.id,
                    dev.name,
                    dev.description,
                    dev.owner_email,
                    dev.created_at.isoformat(),
                    int(dev.is_active),
                ),
            )
            conn.commit()

    @staticmethod
    def _row_to_project(row: sqlite3.Row | tuple[str, str, str, str, str, int]) -> Project:
        return Project(
            id=row[0],
            name=row[1],
            description=row[2] or "",
            owner_email=row[3] or "",
            created_at=row[4],
            is_active=bool(row[5]),
        )
