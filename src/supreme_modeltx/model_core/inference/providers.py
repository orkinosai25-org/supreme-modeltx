"""Optional local pretrained inference and a framework-light provider contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field


class InferenceProvider(Protocol):
    @property
    def is_ready(self) -> bool: ...

    @property
    def metadata(self) -> dict[str, object]: ...

    def generate_from_messages(
        self,
        messages: list,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> tuple[str, int, int]: ...


class PretrainedArtifact(BaseModel):
    """Operator-reviewed provenance for a local, safetensors-only snapshot."""

    source_model: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    conversion_lineage: list[str]
    files_sha256: dict[str, str] = Field(min_length=1)

    @classmethod
    def verify(cls, root: Path) -> "PretrainedArtifact":
        root = root.resolve()
        for path in root.rglob("*"):
            if path.is_symlink() and not path.resolve().is_relative_to(root):
                raise ValueError("Artifact file must remain inside the local snapshot.")
        artifact = cls.model_validate_json((root / "smtx-artifact.json").read_text())
        actual = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path != root / "smtx-artifact.json"
        }
        if actual != set(artifact.files_sha256):
            raise ValueError("Artifact manifest must hash every snapshot file.")
        if not {"config.json", "tokenizer_config.json"} <= actual:
            raise ValueError("Snapshot requires model and tokenizer configuration.")
        if not any(name.endswith(".safetensors") for name in actual):
            raise ValueError("Snapshot requires safetensors weights.")
        if "adapter_config.json" in actual:
            raise ValueError("Use a complete merged snapshot, not an adapter with an external base model.")
        for name, expected in artifact.files_sha256.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Artifact file must remain inside the local snapshot.")
            if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
                raise ValueError("Artifact hashes must be lowercase SHA-256.")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Artifact hash mismatch: {name}")
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("Safetensors index requires a nonempty weight_map.")
            for shard in weight_map.values():
                if not isinstance(shard, str) or shard not in actual or not shard.endswith(".safetensors"):
                    raise ValueError("Every indexed shard must be a hashed local safetensors file.")
        tokenizer_config = json.loads((root / "tokenizer_config.json").read_text())
        if not isinstance(tokenizer_config, dict):
            raise ValueError("Tokenizer configuration must be an object.")
        if tokenizer_config.get("init_inputs"):
            raise ValueError("Tokenizer positional file overrides are not supported.")
        fast_files = tokenizer_config.get("fast_tokenizer_files")
        if fast_files is not None:
            if not isinstance(fast_files, list) or any(
                not isinstance(reference, str) or reference not in actual for reference in fast_files
            ):
                raise ValueError("Tokenizer fast file variants must name hashed snapshot files.")
        for key, reference in tokenizer_config.items():
            if key.endswith("_file") and reference is not None:
                if not isinstance(reference, str) or reference not in actual:
                    raise ValueError("Tokenizer file overrides must name hashed snapshot files.")
        return artifact


class LocalTransformersProvider:
    """Single-request baseline; no downloads, custom model code, or GPU fallback."""

    def __init__(self, model_path: str, device: str = "cpu", dtype: str | None = None) -> None:
        root = Path(model_path).resolve()
        if not root.is_dir():
            raise ValueError("SMTX_PRETRAINED_MODEL_PATH must be a local snapshot directory.")
        self.artifact = PretrainedArtifact.verify(root)

        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device not in {"cpu", "cuda"}:
            raise ValueError("Pretrained inference supports explicit cpu or cuda devices only.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested for pretrained inference but is unavailable.")
        dtype = dtype or ("float32" if device == "cpu" else "bfloat16")
        dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype not in dtypes or (device == "cpu" and dtype != "float32"):
            raise ValueError("Use float32 on CPU, or float32/float16/bfloat16 on CUDA.")
        if device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support BF16.")
        tokenizer_files = {
            key: str(root / reference)
            for key, reference in json.loads((root / "tokenizer_config.json").read_text()).items()
            if key.endswith("_file") and reference is not None
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False,
            **tokenizer_files,
        )
        if not self.tokenizer.chat_template:
            raise ValueError("Pretrained chat inference requires the model's chat template.")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=dtypes[dtype],
        ).to(device).eval()
        self.device = device
        self.dtype = dtype
        self.runtime_versions = {
            "torch": torch.__version__,
            "transformers": getattr(transformers, "__version__", "unknown"),
        }
        self.context_length = int(self.model.config.max_position_embeddings)
        if self.model.config.model_type != self.artifact.architecture:
            raise ValueError("Model architecture does not match the approved artifact manifest.")

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "backend": "transformers",
            "artifact": self.artifact.model_dump(),
            "checkpoint_format": "safetensors",
            "device": self.device,
            "dtype": self.dtype,
            "context_length": self.context_length,
            "runtime_versions": self.runtime_versions,
        }

    def generate_from_messages(
        self,
        messages: list,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> tuple[str, int, int]:
        import torch

        if max_new_tokens < 1 or temperature < 0 or not 0 <= top_p <= 1:
            raise ValueError("Invalid generation limits or sampling parameters.")
        prompt = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        if prompt_tokens + max_new_tokens > self.context_length:
            raise ValueError("Prompt and generation budget exceed the model context window.")
        sampling = {"do_sample": temperature > 0}
        if temperature > 0:
            sampling.update(temperature=temperature, top_p=top_p)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, **sampling,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
            )
        completion = output[0, prompt_tokens:]
        return (
            self.tokenizer.decode(completion, skip_special_tokens=True),
            prompt_tokens,
            int(completion.shape[-1]),
        )
