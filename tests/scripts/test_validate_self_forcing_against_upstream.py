# Copyright 2026 HuggingFace Inc.
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

import importlib.util
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

import torch


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "validate_self_forcing_against_upstream.py"
SPEC = importlib.util.spec_from_file_location("validate_self_forcing_against_upstream", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

EXAMPLE_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "examples" / "inference" / "autoregressive_video_generation.py"
)
EXAMPLE_SPEC = importlib.util.spec_from_file_location("autoregressive_video_generation", EXAMPLE_PATH)
EXAMPLE_MODULE = importlib.util.module_from_spec(EXAMPLE_SPEC)
EXAMPLE_SPEC.loader.exec_module(EXAMPLE_MODULE)


class TestValidateSelfForcingAgainstUpstreamHelpers(unittest.TestCase):
    def test_build_sf_denoising_steps_matches_upstream_schedule(self):
        expected = torch.tensor([1000.0, 937.5, 833.3333129882812, 625.0], dtype=torch.float32)

        steps = MODULE._build_sf_denoising_steps(torch.device("cpu"))

        self.assertTrue(torch.allclose(steps.cpu(), expected))

    def test_example_build_sf_denoising_steps_matches_upstream_schedule(self):
        expected = torch.tensor([1000.0, 937.5, 833.3333129882812, 625.0], dtype=torch.float32)

        steps = EXAMPLE_MODULE._build_sf_denoising_steps(torch.device("cpu"))

        self.assertTrue(torch.allclose(steps.cpu(), expected))

    def test_load_prompt_embeds_from_tensor_payload(self):
        expected = torch.randn(1, 4, 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "prompt_embeds.pt"
            torch.save(expected, path)

            loaded = MODULE._load_prompt_embeds(path)

        self.assertTrue(torch.equal(loaded, expected))

    def test_load_prompt_embeds_from_dict_payload(self):
        expected = torch.randn(1, 4, 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "prompt_embeds.pt"
            torch.save({"prompt_embeds": expected}, path)

            loaded = MODULE._load_prompt_embeds(path)

        self.assertTrue(torch.equal(loaded, expected))

    def test_self_forcing_transformer_guard_rejects_missing_cross_attention_norm(self):
        transformer = SimpleNamespace(
            config=SimpleNamespace(cross_attn_norm=False),
            blocks=[SimpleNamespace(norm2=torch.nn.Identity())],
        )

        with self.assertRaisesRegex(ValueError, "missing Self-Forcing cross-attention norms"):
            MODULE._assert_valid_self_forcing_transformer(transformer)

        with self.assertRaisesRegex(ValueError, "missing Self-Forcing cross-attention norms"):
            EXAMPLE_MODULE._assert_valid_self_forcing_transformer(transformer)


if __name__ == "__main__":
    unittest.main()
