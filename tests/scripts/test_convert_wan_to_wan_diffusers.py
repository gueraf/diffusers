# Copyright 2025 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest

import torch
from safetensors.torch import load_file


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "convert_wan_to_wan_diffusers.py"
SPEC = importlib.util.spec_from_file_location("convert_wan_to_wan_diffusers", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestConvertWanToDiffusersHelpers(unittest.TestCase):
    def test_rename_key_maps_self_attention_weights(self):
        renamed = MODULE.rename_key("blocks.0.self_attn.q.weight")
        self.assertEqual(renamed, "blocks.0.attn1.to_q.weight")

    def test_rename_key_maps_cross_attention_norm_to_diffusers_norm2(self):
        renamed = MODULE.rename_key("blocks.0.norm3.weight")
        self.assertEqual(renamed, "blocks.0.norm2.weight")

    def test_self_forcing_uses_cross_attention_norm(self):
        self.assertTrue(MODULE.WAN_T2V_1_3B_CONFIG["cross_attn_norm"])

    def test_equivalence_report_counts_exact_matches(self):
        reference = {
            "foo": torch.tensor([1.0, 2.0]),
            "bar": torch.tensor([3.0]),
        }
        reloaded = {
            "foo": torch.tensor([1.0, 2.0]),
            "bar": torch.tensor([3.25]),
        }

        report = MODULE._build_equivalence_report(reference, reloaded)

        self.assertEqual(report["num_compared_tensors"], 2)
        self.assertEqual(report["num_exact_tensor_matches"], 1)
        self.assertGreater(report["max_abs_diff"], 0.0)
        self.assertGreater(report["mean_abs_diff"], 0.0)

    def test_convert_self_forcing_checkpoint_keeps_cross_attention_norm_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = pathlib.Path(tmpdir) / "checkpoint.pt"
            output_path = pathlib.Path(tmpdir) / "converted"

            torch.save(
                {
                    "generator_ema": {
                        "model.blocks.0.norm3.weight": torch.ones(1536),
                        "model.blocks.0.norm3.bias": torch.zeros(1536),
                    }
                },
                checkpoint_path,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                MODULE.convert_self_forcing_checkpoint(
                    checkpoint_path=str(checkpoint_path),
                    output_path=str(output_path),
                    use_ema=True,
                    device="cpu",
                )

            with open(output_path / "config.json") as handle:
                config = json.load(handle)
            converted_state_dict = load_file(str(output_path / "diffusion_pytorch_model.safetensors"))

            self.assertTrue(config["cross_attn_norm"])
            self.assertIn("blocks.0.norm2.weight", converted_state_dict)
            self.assertIn("blocks.0.norm2.bias", converted_state_dict)
            self.assertNotIn("blocks.0.norm3.weight", converted_state_dict)


if __name__ == "__main__":
    unittest.main()
