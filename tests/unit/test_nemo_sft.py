"""Synthetic, offline checks only; these tests do not qualify a model or corpus."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4
import sys


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("nemo_sft", ROOT / "training/nemo_sft.py")
sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sft)
CONFIG = ROOT / "configs/nemo/mistral-7b-v03-lora-v1.json"


class NeMoSFTTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "artifacts/private_foundation" / f"nemo-test-{uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("No network"))
        network.start()
        self.addCleanup(network.stop)
        self.config = sft.validate_config(CONFIG)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = {
            "schema_version": 1, "dataset_id": "synthetic-tests-only-v1", "language": "tr",
            "approved_by": "test-fixture-not-a-real-approval",
            "pii_review_passed": True, "secret_review_passed": True,
            "near_duplicate_review_passed": True,
            "base_model": {
                "model_id": sft.MODEL, "revision": "a" * 40,
                "license": "Apache-2.0", "approved_by": "fixture",
            },
            "sources": [
                {
                    "id": split, "group_id": split, "uri": "https://example.invalid/fixture",
                    "license": "test-only", "license_evidence": "not-real",
                    "retrieved_at": "2026-01-01", "content_sha256": "b" * 64,
                    "transform": "synthetic fixture", "reviewed_by": "fixture",
                    "training_allowed": True,
                } for split in ("train", "validation")
            ],
            "splits": {
                "train": {"file": "training.jsonl"},
                "validation": {"file": "validation.jsonl"},
            },
        }
        self.rows = {
            split: [{
                "id": split, "input": f"Test girdisi {split}", "output": f"Test çıktısı {split}",
                "domain": "tourism", "language": "tr", "source_id": split,
                "source_locator": "fixture:1", "reviewed_by": "fixture",
            }] for split in ("train", "validation")
        }

    def write_fixture(self):
        for split, rows in self.rows.items():
            path = self.root / self.manifest["splits"][split]["file"]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.manifest["splits"][split]["sha256"] = sft.digest(path)
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def validate(self):
        self.write_fixture()
        return sft.validate_data(self.manifest_path, self.config)

    def test_valid_config_and_manifest_offline(self):
        _, _, counts = self.validate()
        self.assertEqual(counts["train"]["tourism"], 1)

    def test_invalid_configs(self):
        for key, value in (
            ("method", "full"), ("model_id", "unverified/model"), ("nemo_version", "3.0.0"),
            ("max_steps", 1001), ("warmup_steps", 100), ("global_batch_size", True),
            ("seq_length", 513), ("val_check_interval", 101), ("learning_rate", float("nan")),
        ):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config[key] = value
                path = self.root / "bad-config.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    sft.validate_config(path)

    def test_template_fails_closed(self):
        with self.assertRaises(ValueError):
            sft.validate_data(ROOT / "data/manifests/nemo-sft-v1.template.json", self.config)

    def test_required_approvals(self):
        for flag in ("pii_review_passed", "secret_review_passed", "near_duplicate_review_passed"):
            with self.subTest(flag=flag):
                self.manifest[flag] = False
                with self.assertRaisesRegex(ValueError, "review"):
                    self.validate()
                self.manifest[flag] = True

    def test_unapproved_source(self):
        self.manifest["sources"][0]["training_allowed"] = False
        with self.assertRaisesRegex(ValueError, "rights"):
            self.validate()

    def test_missing_provenance(self):
        del self.manifest["sources"][0]["license_evidence"]
        with self.assertRaisesRegex(ValueError, "license_evidence"):
            self.validate()

    def test_unpinned_model(self):
        self.manifest["base_model"]["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.validate()

    def test_content_tampering(self):
        self.write_fixture()
        (self.root / "training.jsonl").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            sft.validate_data(self.manifest_path, self.config)

    def test_source_group_leakage(self):
        self.manifest["sources"][1]["group_id"] = "train"
        with self.assertRaisesRegex(ValueError, "group leakage"):
            self.validate()

    def test_normalized_prompt_leakage(self):
        self.rows["validation"][0]["input"] = "  TEST GİRDİSİ TRAIN  ".casefold()
        self.rows["train"][0]["input"] = self.rows["validation"][0]["input"].strip()
        with self.assertRaisesRegex(ValueError, "Prompt leakage"):
            self.validate()

    def test_duplicate_pair(self):
        duplicate = copy.deepcopy(self.rows["train"][0])
        duplicate["id"] = "new-id"
        self.rows["train"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate normalized"):
            self.validate()

    def test_religious_review_required_without_generating_religious_text(self):
        self.rows["train"][0]["domain"] = "islamic_knowledge"
        with self.assertRaisesRegex(ValueError, "qualified human"):
            self.validate()

    def test_invalid_record_fields(self):
        for key, value in (("domain", "other"), ("language", "en"), ("output", ""),
                           ("source_id", "unknown"), ("reviewed_by", "")):
            original = self.rows["train"][0][key]
            with self.subTest(key=key):
                self.rows["train"][0][key] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.rows["train"][0][key] = original

    def test_empty_split(self):
        self.rows["validation"] = []
        with self.assertRaisesRegex(ValueError, "empty split"):
            self.validate()

    def test_token_budget_gate(self):
        _, paths, _ = self.validate()
        tokenizer = SimpleNamespace(text_to_ids=lambda value: list(value))
        sft.validate_token_lengths(paths, tokenizer, 512)
        with self.assertRaisesRegex(ValueError, "do not truncate"):
            sft.validate_token_lengths(paths, tokenizer, 16)

    def test_recipe_override_wiring_without_nemo_or_weights(self):
        recipe = SimpleNamespace(
            resume=SimpleNamespace(restore_config=SimpleNamespace(path="nemo://unused")),
            model=SimpleNamespace(config=SimpleNamespace()), trainer=SimpleNamespace(),
            optim=SimpleNamespace(config=SimpleNamespace(), lr_scheduler=SimpleNamespace()),
            peft=SimpleNamespace(), log=SimpleNamespace(ckpt=SimpleNamespace()),
        )
        factory_calls = []

        def factory(**kwargs):
            factory_calls.append(kwargs)
            return recipe

        modules = {
            "nemo_run": SimpleNamespace(
                Config=lambda cls, **kwargs: SimpleNamespace(cls=cls, **kwargs)),
            "nemo.collections.llm.gpt.data.fine_tuning": SimpleNamespace(
                FineTuningDataModule=object),
            "nemo.collections.llm.recipes": SimpleNamespace(
                mistral_7b=SimpleNamespace(finetune_recipe=factory)),
        }
        with patch.dict(sys.modules, modules):
            built = sft.build_recipe(self.config, self.root, self.root / "base", self.root / "out")
        self.assertIs(built, recipe)
        self.assertEqual(factory_calls[0]["num_gpus_per_node"], 1)
        self.assertEqual(factory_calls[0]["peft_scheme"], "lora")
        self.assertFalse(factory_calls[0]["packed_sequence"])
        self.assertEqual(recipe.resume.restore_config.path, str(self.root / "base"))
        self.assertFalse(recipe.resume.resume_if_exists)
        self.assertEqual(recipe.data.dataset_kwargs["prompt_template"], sft.PROMPT_TEMPLATE)
        self.assertTrue(recipe.data.dataset_kwargs["answer_only_loss"])
        self.assertEqual(recipe.data.global_batch_size, self.config["global_batch_size"])
        self.assertEqual(recipe.trainer.max_steps, self.config["max_steps"])
        self.assertEqual(recipe.optim.lr_scheduler.max_steps, self.config["max_steps"])
        self.assertEqual(recipe.optim.lr_scheduler.warmup_steps, self.config["warmup_steps"])
        self.assertEqual(recipe.peft.dim, self.config["lora_rank"])
        self.assertEqual(recipe.log.ckpt.every_n_train_steps, self.config["val_check_interval"])
        self.assertIsNone(recipe.log.wandb)

    def test_train_requires_manifest_before_runtime_imports(self):
        with self.assertRaises(SystemExit) as result:
            sft.main(["--config", str(CONFIG), "--train"])
        self.assertEqual(result.exception.code, 2)

    def test_runtime_version_gate_before_gpu_import(self):
        manifest, paths, _ = self.validate()
        with patch.object(sft, "version", return_value="3.0.0"), patch.dict(sft.os.environ):
            with self.assertRaisesRegex(ValueError, "version mismatch"):
                sft.train(self.config, manifest, paths, self.root / "base", self.root / "out")

    def prepare_base_receipt(self):
        base = self.root / "base"
        (base / "context").mkdir(parents=True)
        (base / "weights").mkdir()
        (base / "smtx-base.json").write_text(json.dumps({
            "nemo_version": "2.4.0", "base_model": self.manifest["base_model"],
        }), encoding="utf-8")
        return base

    def test_runtime_base_identity_gate_before_gpu_import(self):
        manifest, paths, _ = self.validate()
        base = self.prepare_base_receipt()
        manifest = copy.deepcopy(manifest)
        manifest["base_model"]["revision"] = "c" * 40
        with patch.object(sft, "version", return_value="2.4.0"), patch.dict(sft.os.environ):
            with self.assertRaisesRegex(ValueError, "Checkpoint receipt"):
                sft.train(self.config, manifest, paths, base, self.root / "out")

    def test_runtime_refuses_existing_output_before_gpu_import(self):
        manifest, paths, _ = self.validate()
        base = self.prepare_base_receipt()
        output = self.root / "out"
        output.mkdir()
        with patch.object(sft, "version", return_value="2.4.0"), patch.dict(sft.os.environ):
            with self.assertRaisesRegex(ValueError, "fresh output"):
                sft.train(self.config, manifest, paths, base, output)


if __name__ == "__main__":
    unittest.main()
