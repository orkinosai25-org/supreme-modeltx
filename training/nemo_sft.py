"""Offline gates and an opt-in, single-GPU NeMo 2.4 Mistral LoRA pilot."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import unicodedata


DOMAINS = {"turkish_culture", "tourism", "islamic_knowledge"}
MODEL = "mistralai/Mistral-7B-v0.3"
PROMPT_TEMPLATE = "Soru: {input}\nYanıt: {output}"
CONFIG_KEYS = {
    "schema_version", "model_id", "nemo_version", "method", "domains", "seed",
    "seq_length", "micro_batch_size", "global_batch_size", "max_steps",
    "val_check_interval", "limit_val_batches", "learning_rate", "warmup_steps",
    "lora_rank", "lora_alpha", "lora_dropout",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        result = json.load(handle)
    require(isinstance(result, dict), f"{path}: expected an object")
    return result


def text(value):
    return isinstance(value, str) and bool(value.strip())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def validate_config(path):
    config = load_json(path)
    require(set(config) == CONFIG_KEYS, "Unexpected or missing config keys")
    for key, expected in {
        "schema_version": 1, "model_id": MODEL, "nemo_version": "2.4.0",
        "method": "lora", "micro_batch_size": 1,
    }.items():
        require(config[key] == expected, f"Unsupported {key}")
    require(isinstance(config["domains"], list)
            and len(config["domains"]) == 3 and set(config["domains"]) == DOMAINS,
            "Expected the three approved domain identifiers")
    for key in ("seed", "seq_length", "global_batch_size", "max_steps",
                "val_check_interval", "limit_val_batches", "warmup_steps",
                "lora_rank", "lora_alpha"):
        require(type(config[key]) is int and config[key] > 0, f"Invalid {key}")
    require(config["seq_length"] <= 2048 and config["seq_length"] % 16 == 0,
            "seq_length must be a multiple of 16, at most 2048 for this pilot")
    require(config["max_steps"] <= 1000, "Pilot limited to 1000 steps")
    require(config["warmup_steps"] < config["max_steps"], "Warmup exceeds run")
    require(config["val_check_interval"] <= config["max_steps"],
            "Run must include validation")
    for key, low, high in (("learning_rate", 0, 0.001), ("lora_dropout", -1e-10, 0.5)):
        require(type(config[key]) in (int, float)
                and low < config[key] <= high, f"Invalid {key}")
    return config


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def validate_data(manifest_path, config):
    manifest_path = Path(manifest_path).resolve()
    manifest = load_json(manifest_path)
    require(manifest.get("schema_version") == 1 and manifest.get("language") == "tr",
            "Expected manifest schema 1 and language tr")
    require(text(manifest.get("dataset_id")) and text(manifest.get("approved_by")),
            "Dataset version and approval required")
    for flag in ("pii_review_passed", "secret_review_passed", "near_duplicate_review_passed"):
        require(manifest.get(flag) is True, f"Required review missing: {flag}")
    base = manifest.get("base_model", {})
    require(isinstance(base, dict), "base_model must be an object")
    require(base.get("model_id") == config["model_id"]
            and re.fullmatch(r"[0-9a-f]{40}", base.get("revision", "")) is not None
            and base.get("license") == "Apache-2.0" and text(base.get("approved_by")),
            "Approve the exact base model, license and immutable HF commit")
    require(isinstance(manifest.get("sources"), list) and manifest["sources"],
            "Approved sources required; template is intentionally not trainable")
    sources = {}
    for source in manifest["sources"]:
        require(isinstance(source, dict), "Source must be an object")
        for key in ("id", "group_id", "uri", "license", "license_evidence",
                    "retrieved_at", "content_sha256", "transform", "reviewed_by"):
            require(text(source.get(key)), f"Missing source {key}")
        require(source.get("training_allowed") is True, "Source training rights not approved")
        require(re.fullmatch(r"[0-9a-f]{64}", source["content_sha256"]) is not None,
                "Invalid source content_sha256")
        require(source["id"] not in sources, "Duplicate source ID")
        sources[source["id"]] = source
    require(set(manifest.get("splits", {})) == {"train", "validation"},
            "Exactly train and validation splits required; keep test data separate")
    seen_ids, seen_prompts, seen_pairs, seen_groups = {}, {}, {}, {}
    paths, counts = {}, {}
    for split, expected_file in (("train", "training.jsonl"), ("validation", "validation.jsonl")):
        entry = manifest["splits"][split]
        require(isinstance(entry, dict), "Split entry must be an object")
        require(entry.get("file") == expected_file, f"Expected {expected_file}")
        path = manifest_path.parent / expected_file
        require(path.resolve().parent == manifest_path.parent, "Split must stay beside manifest")
        require(digest(path) == entry.get("sha256"), f"{split}: SHA-256 mismatch")
        counts[split] = Counter()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                require(isinstance(row, dict), f"{split}:{line_number}: expected object")
                for key in ("id", "input", "output", "source_id", "source_locator", "reviewed_by"):
                    require(text(row.get(key)), f"{split}:{line_number}: missing {key}")
                require(row.get("domain") in DOMAINS, "Unknown domain")
                require(row.get("language") == "tr", "Record must declare language tr")
                require(row["source_id"] in sources, "Unknown source_id")
                source = sources[row["source_id"]]
                if row["domain"] == "islamic_knowledge":
                    require(text(row.get("religious_reviewed_by")),
                            "Islamic knowledge requires a qualified human reviewer")
                prompt = normalized(row["input"])
                pair = (prompt, normalized(row["output"]))
                require(row["id"] not in seen_ids, "Duplicate record ID")
                require(pair not in seen_pairs, "Duplicate normalized input/output")
                require(prompt not in seen_prompts or seen_prompts[prompt] == split,
                        "Prompt leakage across splits")
                group = source["group_id"]
                require(group not in seen_groups or seen_groups[group] == split,
                        "Source/document group leakage across splits")
                seen_ids[row["id"]] = split
                seen_pairs[pair] = split
                seen_prompts[prompt] = split
                seen_groups[group] = split
                counts[split][row["domain"]] += 1
        require(sum(counts[split].values()) > 0, f"{split}: empty split")
        paths[split] = path
    return manifest, paths, counts


def build_recipe(config, dataset_root, base_checkpoint, output):
    import nemo_run as run
    from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule
    from nemo.collections.llm.recipes import mistral_7b

    recipe = mistral_7b.finetune_recipe(
        dir=str(output), name="mistral-tr-sft-v1", num_nodes=1,
        num_gpus_per_node=1, peft_scheme="lora", packed_sequence=False,
    )
    recipe.resume.restore_config.path = str(base_checkpoint)
    recipe.resume.resume_if_exists = False
    recipe.data = run.Config(
        FineTuningDataModule, dataset_root=str(dataset_root),
        seq_length=config["seq_length"], micro_batch_size=config["micro_batch_size"],
        global_batch_size=config["global_batch_size"], seed=config["seed"],
        num_workers=0, memmap_workers=1,
        dataset_kwargs={
            "prompt_template": PROMPT_TEMPLATE,
            "label_key": "output", "answer_only_loss": True,
            "add_bos": True, "add_eos": True,
        },
    )
    # No FP8 or quantization: keep the frozen base in BF16.
    recipe.model.config.seq_length = config["seq_length"]
    recipe.model.config.recompute_granularity = "full"
    recipe.model.config.recompute_method = "uniform"
    recipe.model.config.recompute_num_layers = 1
    for key in ("max_steps", "val_check_interval", "limit_val_batches"):
        setattr(recipe.trainer, key, config[key])
    recipe.trainer.limit_test_batches = 0
    recipe.trainer.num_sanity_val_steps = 0
    recipe.optim.config.lr = config["learning_rate"]
    recipe.optim.lr_scheduler.max_steps = config["max_steps"]
    recipe.optim.lr_scheduler.warmup_steps = config["warmup_steps"]
    recipe.peft.dim = config["lora_rank"]
    recipe.peft.alpha = config["lora_alpha"]
    recipe.peft.dropout = config["lora_dropout"]
    recipe.log.ckpt.every_n_train_steps = config["val_check_interval"]
    recipe.log.ckpt.save_last = True
    recipe.log.wandb = None
    return recipe


def validate_token_lengths(paths, tokenizer, seq_length):
    for split, path in paths.items():
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                row = json.loads(line)
                # Match v2.4 GPTSFTDataset._separate_template, including whitespace handling.
                space = " " if getattr(tokenizer, "space_sensitive", False) else ""
                length = sum(len(tokenizer.text_to_ids(part)) for part in (
                    "Soru:", space + row["input"].strip(" "),
                    "\nYanıt:", space + row["output"].strip(" "),
                )) + 2
                require(length <= seq_length,
                        f"{split}:{number}: over token budget; shorten and re-review, do not truncate")


def train(config, manifest, paths, base_checkpoint, output):
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    require(version("nemo_toolkit") == config["nemo_version"], "NeMo version mismatch")
    base_checkpoint = Path(base_checkpoint).resolve()
    require((base_checkpoint / "context").is_dir()
            and (base_checkpoint / "weights").is_dir(), "Expected converted NeMo 2 checkpoint")
    receipt = load_json(base_checkpoint / "smtx-base.json")
    require(receipt.get("base_model") == manifest["base_model"]
            and receipt.get("nemo_version") == config["nemo_version"],
            "Checkpoint receipt must match approved base model and NeMo version")
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh output directory; resumption is not supported")
    require(output != base_checkpoint and base_checkpoint not in output.parents,
            "Output must not be inside base checkpoint")
    import torch
    import nemo_run as run
    from lightning.pytorch import seed_everything
    from nemo.lightning import io

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Expose exactly one approved CUDA GPU")
    require(torch.cuda.is_bf16_supported(), "BF16-capable GPU required")
    tokenizer = io.load_context(base_checkpoint / "context", "model.tokenizer")
    validate_token_lengths(paths, tokenizer, config["seq_length"])
    seed_everything(config["seed"], workers=True)
    recipe = build_recipe(config, paths["train"].parent, base_checkpoint, output)
    output.mkdir(parents=True)
    (output / "smtx-run.json").write_text(json.dumps({
        "config": config, "manifest": manifest, "base_checkpoint": str(base_checkpoint),
        "artifact": "NeMo LoRA adapter; frozen base required",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run.run(recipe, direct=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--train", action="store_true", help="Explicitly opt in to GPU training")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        config = validate_config(args.config)
        if not args.manifest:
            require(not args.train, "--train requires --manifest")
            print("Config valid (offline); no data, model, GPU or quality validation performed.")
            return
        manifest, paths, counts = validate_data(args.manifest, config)
        print(json.dumps({"offline_preflight": "passed", "domain_counts": counts}))
        if args.train:
            require(args.base_checkpoint and args.output,
                    "--train requires --base-checkpoint and --output")
            train(config, manifest, paths, args.base_checkpoint, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Preflight failed: {error}\n")


if __name__ == "__main__":
    main()
