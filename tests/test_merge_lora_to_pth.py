"""Exercise the CLI exporter on a tiny CPU GPT; no downloaded models needed.

Run from the repository root with:
    python -m unittest discover -s tests -p test_merge_lora_to_pth.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2Model

from tools import merge_lora_to_pth as exporter


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MergeLoRAToPthTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tiny-lora-export-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "base.pth"
        self.expected_base = self.root / "deployment-base.pth"
        self.config = self.root / "config.yaml"
        self.adapter = self.root / "adapter"
        self.output = self.root / "export" / "merged.pth"
        self.config.write_text("gpt:\n  heads: 2\n", encoding="utf-8")

        # Preserve the caller's RNG state; all fixture tensors are deterministic.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73)
            tiny = GPT2Model(GPT2Config(
                vocab_size=256, n_positions=64, n_ctx=64, n_embd=16,
                n_layer=1, n_head=2, use_cache=False,
                attn_implementation="eager",
            )).eval()
            self.state = {
                f"gpt.{name}": tensor.detach().clone()
                for name, tensor in tiny.state_dict().items()
                if name not in {"wte.weight", "wpe.weight"}
            }
            self.state.update({
                "mel_pos_embedding.emb.weight": torch.randn(32, 16),
                "text_pos_embedding.emb.weight": torch.randn(32, 16),
                "frozen_projection.weight": torch.randn(4, 16),
            })
            torch.save(self.state, self.base)
            shutil.copyfile(self.base, self.expected_base)

            trunk = exporter.make_gpt(self.state, {"heads": 2})
            self.live_adapter = get_peft_model(trunk, LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.0, bias="none",
                target_modules=["c_attn", "c_proj", "c_fc"],
            )).eval()
            with torch.no_grad():
                for name, parameter in self.live_adapter.named_parameters():
                    if ".lora_" in name:
                        parameter.copy_(torch.randn_like(parameter) * 0.08)
            self.live_adapter.save_pretrained(self.adapter, safe_serialization=True)

        self.source_hashes = {
            path: sha256(path)
            for path in (self.base, self.expected_base, self.config,
                         self.adapter / "adapter_config.json",
                         self.adapter / "adapter_model.safetensors")
        }

    def run_export(self, *, output=None, expected_base=None):
        arguments = [
            "merge_lora_to_pth.py", "--base", str(self.base),
            "--config", str(self.config), "--adapter", str(self.adapter),
            "--output", str(output or self.output),
            "--expected-base", str(expected_base or self.expected_base),
        ]
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            exporter.main()

    def assert_sources_unchanged(self):
        for path, before in self.source_hashes.items():
            self.assertEqual(sha256(path), before, str(path))

    def test_merge_round_trip_matches_live_adapter_and_changes_only_lora_targets(self):
        self.run_export()
        merged = torch.load(self.output, map_location="cpu", weights_only=True)
        self.assertEqual(set(merged), set(self.state))
        adapter_state = load_file(str(self.adapter / "adapter_model.safetensors"))
        target_names = {
            "gpt." + name.removeprefix("base_model.model.").removesuffix(".lora_A.weight") + ".weight"
            for name in adapter_state if name.endswith(".lora_A.weight")
        }
        self.assertEqual(len(target_names), 4)
        changed = {name for name in self.state if not torch.equal(merged[name], self.state[name])}
        self.assertEqual(changed, target_names)
        for name, value in merged.items():
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(value.shape, self.state[name].shape)
            self.assertTrue(bool(torch.isfinite(value).all()), name)
            self.assertNotIn("lora_", name)
            self.assertNotIn("base_model.", name)
            self.assertNotIn("inference_model.", name)
            if name not in target_names:
                self.assertTrue(torch.equal(value, self.state[name]), name)

        # Independent Conv1D LoRA calculation catches a dropped or transposed delta.
        for name in target_names:
            prefix = "base_model.model." + name.removeprefix("gpt.").removesuffix(".weight")
            delta = (adapter_state[prefix + ".lora_B.weight"] @
                     adapter_state[prefix + ".lora_A.weight"]).T * (4 / 2)
            torch.testing.assert_close(merged[name], self.state[name] + delta)

        probe = torch.randn(1, 9, 16, generator=torch.Generator().manual_seed(91))
        restored = exporter.make_gpt(merged, {"heads": 2})
        with torch.inference_mode():
            expected = self.live_adapter(inputs_embeds=probe, use_cache=False).last_hidden_state
            actual = restored(inputs_embeds=probe, use_cache=False).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)

        report = json.loads(self.output.with_suffix(".merge.json").read_text(encoding="utf-8"))
        self.assertEqual(report["base_sha256"], self.source_hashes[self.base])
        self.assertEqual(report["adapter_sha256"], self.source_hashes[self.adapter / "adapter_model.safetensors"])
        self.assertEqual(report["output_sha256"], sha256(self.output))
        self.assertEqual(Path(report["output"]), self.output.resolve())
        self.assertEqual(report["dtype"], "float32")
        self.assertEqual(report["tensor_count"], len(self.state))
        self.assertEqual(report["lora_tensor_count"], len(adapter_state))
        self.assertEqual(report["lora_target_count"], len(target_names))
        self.assertEqual(report["changed_tensor_count"], len(target_names))
        self.assertEqual(set(report["changed_tensor_names"]), target_names)
        self.assertEqual((report["rank"], report["alpha"]), (2, 4))
        self.assertLessEqual(report["cpu_forward_max_abs_difference"], 2e-4)
        self.assertTrue(report["source_files_unchanged"])
        self.assertTrue(report["round_trip_tensors_identical"])
        self.assertFalse(self.output.with_suffix(".pth.partial").exists())
        self.assert_sources_unchanged()

    def test_refuses_overwriting_existing_exports_or_sources(self):
        existing = self.root / "existing.pth"
        existing.write_bytes(b"keep this export")
        for path in (existing, self.base):
            with self.subTest(output=path), self.assertRaises(FileExistsError):
                self.run_export(output=path)
        self.assertEqual(existing.read_bytes(), b"keep this export")
        with self.assertRaisesRegex(ValueError, "separate from the base and adapter"):
            self.run_export(output=self.adapter / "merged.pth")
        self.assertFalse((self.adapter / "merged.pth").exists())
        self.assert_sources_unchanged()

    def test_refuses_different_deployment_base_without_creating_export(self):
        different = self.root / "different-base.pth"
        altered = {name: tensor.clone() for name, tensor in self.state.items()}
        altered["frozen_projection.weight"][0, 0] += 1.0
        torch.save(altered, different)
        with self.assertRaisesRegex(ValueError, "Training base and deployment base differ"):
            self.run_export(expected_base=different)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".merge.json").exists())
        self.assertFalse(self.output.with_suffix(".pth.partial").exists())
        self.assert_sources_unchanged()


if __name__ == "__main__":
    unittest.main()
