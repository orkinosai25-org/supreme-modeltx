"""
schemas.py — SUMOTX Platform API request/response models (v0.1)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Common helpers ─────────────────────────────────────────────────────────────


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Auth ───────────────────────────────────────────────────────────────────────


class TokenRequest(BaseModel):
    api_key: str = Field(..., description="API key issued by SUMOTX.")

    model_config = {
        "json_schema_extra": {
            "examples": [{"api_key": "smtx_sk_live_xxxxxxxxxxxx"}]
        }
    }


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="Bearer token for subsequent requests.")
    token_type: str = Field(default="bearer")
    expires_in: int = Field(default=3600, description="Token lifetime in seconds.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "bearer",
                    "expires_in": 3600,
                }
            ]
        }
    }


# ── Projects ───────────────────────────────────────────────────────────────────


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="Human-readable project name.")
    description: Optional[str] = Field(None, max_length=512)
    tags: Dict[str, str] = Field(default_factory=dict, description="Arbitrary key-value metadata.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "my-llm-project",
                    "description": "Fine-tuning T-101 on customer support data.",
                    "tags": {"team": "ml", "env": "dev"},
                }
            ]
        }
    }


class Project(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    description: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "proj_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "name": "my-llm-project",
                    "description": "Fine-tuning T-101 on customer support data.",
                    "tags": {"team": "ml", "env": "dev"},
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-01T00:00:00Z",
                }
            ]
        }
    }


class ProjectList(BaseModel):
    items: List[Project]
    total: int


# ── Deployments ────────────────────────────────────────────────────────────────


class DeploymentCreate(BaseModel):
    project_id: str = Field(..., description="Target project identifier.")
    model_id: str = Field(..., description="Model to deploy (from the model registry).")
    azure_region: str = Field(default="eastus", description="Azure region for provisioning.")
    vm_sku: str = Field(default="Standard_NC6s_v3", description="GPU VM SKU for inference.")
    min_replicas: int = Field(default=1, ge=1, le=10)
    max_replicas: int = Field(default=3, ge=1, le=50)
    environment: str = Field(default="dev", pattern="^(dev|staging|prod)$")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "project_id": "proj_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "model_id": "model_t101_v1",
                    "azure_region": "eastus",
                    "vm_sku": "Standard_NC6s_v3",
                    "min_replicas": 1,
                    "max_replicas": 3,
                    "environment": "prod",
                }
            ]
        }
    }


class DeploymentStatus(str, Enum):
    pending = "pending"
    provisioning = "provisioning"
    running = "running"
    failed = "failed"
    stopped = "stopped"


class Deployment(BaseModel):
    id: str = Field(default_factory=_new_id)
    project_id: str
    model_id: str
    azure_region: str
    vm_sku: str
    min_replicas: int
    max_replicas: int
    environment: str
    status: DeploymentStatus = DeploymentStatus.pending
    endpoint_url: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class DeploymentStatusResponse(BaseModel):
    id: str
    status: DeploymentStatus
    endpoint_url: Optional[str] = None
    message: Optional[str] = None
    updated_at: datetime = Field(default_factory=_now)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "depl_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "status": "running",
                    "endpoint_url": "https://smtx-prod-orchestrator.azurewebsites.net",
                    "message": "Deployment healthy — 2/2 replicas ready.",
                    "updated_at": "2025-01-01T00:05:00Z",
                }
            ]
        }
    }


# ── Training runs ──────────────────────────────────────────────────────────────


class TrainingRunCreate(BaseModel):
    project_id: str = Field(..., description="Target project identifier.")
    base_model: str = Field(default="tmodels/t101", description="Base model path or registry ID.")
    dataset_path: str = Field(..., description="Azure Blob Storage path to the training dataset.")
    num_epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=8, ge=1, le=256)
    learning_rate: float = Field(default=2e-5, gt=0.0)
    warmup_steps: int = Field(default=100, ge=0)
    output_model_name: Optional[str] = Field(None, description="Name under which to register the fine-tuned model.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "project_id": "proj_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "base_model": "tmodels/t101",
                    "dataset_path": "az://smtx-data/training/customer-support-v2.jsonl",
                    "num_epochs": 3,
                    "batch_size": 8,
                    "learning_rate": 2e-5,
                    "warmup_steps": 100,
                    "output_model_name": "t101-customer-support-v1",
                }
            ]
        }
    }


class TrainingRunStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TrainingRun(BaseModel):
    id: str = Field(default_factory=_new_id)
    project_id: str
    base_model: str
    dataset_path: str
    num_epochs: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    output_model_name: Optional[str] = None
    status: TrainingRunStatus = TrainingRunStatus.queued
    azure_batch_job_id: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class TrainingRunList(BaseModel):
    items: List[TrainingRun]
    total: int


class TrainingLogEntry(BaseModel):
    timestamp: datetime
    level: str = Field(..., description="Log level: INFO, WARNING, ERROR.")
    message: str


class TrainingRunLogs(BaseModel):
    run_id: str
    logs: List[TrainingLogEntry]


# ── Model registry ─────────────────────────────────────────────────────────────


class ModelStage(str, Enum):
    experimental = "experimental"
    staging = "staging"
    production = "production"
    archived = "archived"


class ModelInfo(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    version: str
    base_model: Optional[str] = None
    project_id: Optional[str] = None
    stage: ModelStage = ModelStage.experimental
    description: Optional[str] = None
    parameters: Optional[int] = Field(None, description="Total parameter count.")
    artifact_path: Optional[str] = Field(None, description="Azure Blob Storage path to model weights.")
    created_at: datetime = Field(default_factory=_now)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "model_t101_v1",
                    "name": "t101-customer-support-v1",
                    "version": "1.0.0",
                    "base_model": "tmodels/t101",
                    "project_id": "proj_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "stage": "staging",
                    "description": "T-101 fine-tuned on customer support corpus.",
                    "parameters": 7000000000,
                    "artifact_path": "az://smtx-models/t101-customer-support-v1/",
                    "created_at": "2025-01-02T12:00:00Z",
                }
            ]
        }
    }


class ModelList(BaseModel):
    items: List[ModelInfo]
    total: int


class PromoteRequest(BaseModel):
    stage: ModelStage = Field(..., description="Target stage to promote the model to.")

    model_config = {
        "json_schema_extra": {
            "examples": [{"stage": "production"}]
        }
    }


# ── Governed model assignment ───────────────────────────────────────────────────


class DataSensitivity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    restricted = "restricted"


class ModelUseCase(str, Enum):
    general_purpose = "general_purpose"
    retrieval = "retrieval"
    verification = "verification"
    orchestration = "orchestration"


class IndexingProfile(str, Enum):
    keyword_basic = "keyword_basic"
    vector_balanced = "vector_balanced"
    hybrid_enterprise = "hybrid_enterprise"


class RagConfiguration(BaseModel):
    enabled: bool = Field(
        ...,
        description="Enable retrieval-augmented generation for this assignment.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Maximum retrieved chunks per query when RAG is enabled.",
    )


class GovernedModelInfo(BaseModel):
    """Model registry entry extended with enterprise governance metadata."""

    id: str
    name: str
    version: str
    stage: ModelStage
    description: Optional[str] = None
    parameters: Optional[int] = None
    # Governance fields
    is_approved: bool = Field(..., description="Admin has explicitly approved this model for enterprise use.")
    benchmark_score: Optional[float] = Field(
        None,
        description="Informational benchmark score (0–100). Does not override governance approval.",
    )
    max_data_sensitivity: DataSensitivity = Field(
        DataSensitivity.low,
        description="Highest data-sensitivity category this model is cleared to handle.",
    )
    supported_use_cases: List[ModelUseCase] = Field(
        default_factory=list,
        description="Use cases this model is cleared to serve.",
    )
    created_at: datetime = Field(default_factory=_now)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "model_t101_base",
                    "name": "T-101",
                    "version": "0.1.0",
                    "stage": "production",
                    "description": "SUMOTX base inference model (7B parameters).",
                    "parameters": 7_000_000_000,
                    "is_approved": True,
                    "benchmark_score": 72.5,
                    "max_data_sensitivity": "high",
                    "supported_use_cases": ["general_purpose"],
                    "created_at": "2025-01-01T00:00:00Z",
                }
            ]
        }
    }


class GovernedModelList(BaseModel):
    items: List[GovernedModelInfo]
    total: int


class ModelAssignmentRequest(BaseModel):
    """Request to assign an approved model to a workspace under enterprise policy."""

    workspace_id: str = Field(..., description="Workspace or project scope for this assignment.")
    tenant_id: str = Field(..., description="Tenant (organisation) scope for this assignment.")
    model_id: str = Field(..., description="ID of the approved model to assign.")
    use_case: ModelUseCase = Field(..., description="Enterprise use case this assignment covers.")
    data_sensitivity: DataSensitivity = Field(
        ..., description="Maximum data-sensitivity level for this workspace."
    )
    data_source_id: str = Field(
        ..., description="Enterprise data source bound to this model assignment."
    )
    indexing_profile: IndexingProfile = Field(
        ...,
        description="Indexing profile used for data-source retrieval in this assignment.",
    )
    rag: RagConfiguration = Field(
        ...,
        description="RAG configuration linked to this data/model assignment policy.",
    )
    assigned_by: str = Field(..., description="Identity of the admin making the assignment.")
    reason: str = Field(
        ...,
        min_length=10,
        description="Mandatory justification for the assignment (audit requirement).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "workspace_id": "ws-finance-reporting",
                    "tenant_id": "tenant-acme-corp",
                    "model_id": "model_t101_base",
                    "use_case": "general_purpose",
                    "data_sensitivity": "high",
                    "data_source_id": "ds_finance_docs",
                    "indexing_profile": "hybrid_enterprise",
                    "rag": {"enabled": False, "top_k": 5},
                    "assigned_by": "admin@acme.corp",
                    "reason": "Approved for GDPR-compliant financial reporting queries after security review SR-2025-042.",
                }
            ]
        }
    }


class ModelAssignmentResponse(BaseModel):
    """Active model assignment for a workspace scope."""

    id: str = Field(default_factory=_new_id)
    workspace_id: str
    tenant_id: str
    model_id: str
    model_name: str
    use_case: ModelUseCase
    data_sensitivity: DataSensitivity
    data_source_id: str
    indexing_profile: IndexingProfile
    rag: RagConfiguration
    assigned_by: str
    assigned_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ModelAssignmentAuditEntry(BaseModel):
    """Immutable audit record for a model assignment change."""

    id: str = Field(default_factory=_new_id)
    workspace_id: str
    tenant_id: str
    use_case: ModelUseCase
    previous_model_id: Optional[str] = Field(
        None, description="Model that was previously assigned (null on first assignment)."
    )
    new_model_id: str
    data_source_id: str
    indexing_profile: IndexingProfile
    rag: RagConfiguration
    changed_by: str
    changed_at: datetime = Field(default_factory=_now)
    reason: str


class ModelAssignmentAuditList(BaseModel):
    workspace_id: str
    tenant_id: str
    entries: List[ModelAssignmentAuditEntry]
    total: int


# ── Chat / Inference ───────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$", description="Message author role.")
    content: str = Field(..., description="Message text content.")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="t101", description="Model identifier to use for completion.")
    messages: List[ChatMessage] = Field(..., min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = Field(default=False, description="Streaming is reserved for a future release.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "t101",
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": "Explain transformer architecture in two sentences."},
                    ],
                    "max_tokens": 256,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "stream": False,
                }
            ]
        }
    }


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=_new_id)
    object: str = Field(default="chat.completion")
    created: int = Field(description="Unix timestamp of creation.")
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "chatcmpl_a1b2c3d4",
                    "object": "chat.completion",
                    "created": 1704067200,
                    "model": "t101",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "A transformer uses self-attention to weigh token relationships...",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 42, "total_tokens": 72},
                }
            ]
        }
    }


# ── Health ─────────────────────────────────────────────────────────────────────


class ServiceHealth(BaseModel):
    name: str
    status: str = Field(..., description="ok | degraded | unreachable")
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall platform status: ok | degraded.")
    version: str
    services: List[ServiceHealth]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ok",
                    "version": "0.1.0",
                    "services": [
                        {"name": "api", "status": "ok", "latency_ms": 1.2},
                        {"name": "inference", "status": "ok", "latency_ms": 45.3},
                        {"name": "retrieval", "status": "ok", "latency_ms": 12.1},
                        {"name": "verification", "status": "ok", "latency_ms": 18.7},
                    ],
                }
            ]
        }
    }


# ── Error ──────────────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"error": {"code": "not_found", "message": "Project not found.", "field": "project_id"}}
            ]
        }
    }
