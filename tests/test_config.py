import json
import tempfile
import unittest

from src.supreme_modeltx.config import TrainingConfig, load_config


class TestTrainingConfig(unittest.TestCase):
    def test_load_default_config(self):
        cfg = load_config()
        self.assertEqual(cfg.num_layers, 6)
        self.assertEqual(cfg.num_heads, 6)

    def test_override_config_from_json(self):
        data = {"batch_size": 4, "max_steps": 3}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(data, fh)
            file_path = fh.name

        cfg = load_config(file_path)
        self.assertEqual(cfg.batch_size, 4)
        self.assertEqual(cfg.max_steps, 3)

    def test_unknown_key_rejected(self):
        data = {"unknown": 1}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(data, fh)
            file_path = fh.name

        with self.assertRaises(ValueError):
            load_config(file_path)


if __name__ == "__main__":
    unittest.main()
