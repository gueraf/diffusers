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
from torch import nn


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

    def test_scheduler_sigma_lookup_matches_upstream_shifted_sigmas(self):
        expected = torch.tensor([1.0, 0.9375, 0.8333333333333334, 0.6250], dtype=torch.float32)
        timesteps = torch.tensor([1000.0, 937.5, 833.3333129882812, 625.0], dtype=torch.float32)

        module_timesteps, module_sigmas = MODULE._build_sf_scheduler_tables(torch.device("cpu"))
        example_timesteps, example_sigmas = EXAMPLE_MODULE._build_sf_scheduler_tables(torch.device("cpu"))

        torch.testing.assert_close(MODULE._lookup_sf_sigma(timesteps, module_timesteps, module_sigmas), expected)
        torch.testing.assert_close(EXAMPLE_MODULE._lookup_sf_sigma(timesteps, example_timesteps, example_sigmas), expected)

    def test_convert_sf_flow_to_x0_uses_scheduler_sigma_lookup(self):
        flow_pred = torch.ones(1, 2, 3, 1, 1, dtype=torch.bfloat16)
        xt = torch.zeros_like(flow_pred)
        timestep = torch.full((1, 3), 937.5, dtype=torch.float32)

        module_timesteps, module_sigmas = MODULE._build_sf_scheduler_tables(torch.device("cpu"))
        example_timesteps, example_sigmas = EXAMPLE_MODULE._build_sf_scheduler_tables(torch.device("cpu"))
        expected = torch.full_like(flow_pred, -0.9375)

        module_x0 = MODULE._convert_sf_flow_to_x0(flow_pred, xt, timestep, module_timesteps, module_sigmas)
        example_x0 = EXAMPLE_MODULE._convert_sf_flow_to_x0(flow_pred, xt, timestep, example_timesteps, example_sigmas)

        torch.testing.assert_close(module_x0.float(), expected.float())
        torch.testing.assert_close(example_x0.float(), expected.float())

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

    def test_align_self_forcing_transformer_dtype_casts_fp32_runtime_modules(self):
        class DummyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, 4, dtype=torch.float32))
                self.norm2 = nn.LayerNorm(4, elementwise_affine=True)

        class DummyConditionEmbedder(nn.Module):
            def __init__(self):
                super().__init__()
                self.time_embedder = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 4))

        class DummyTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embedding = nn.Conv3d(1, 1, kernel_size=1, bias=False).to(dtype=torch.bfloat16)
                self.condition_embedder = DummyConditionEmbedder()
                self.scale_shift_table = nn.Parameter(torch.zeros(1, 2, 4, dtype=torch.float32))
                self.blocks = nn.ModuleList([DummyBlock()])

        transformer = DummyTransformer()
        self.assertEqual(transformer.patch_embedding.weight.dtype, torch.bfloat16)
        self.assertEqual(transformer.condition_embedder.time_embedder[0].weight.dtype, torch.float32)
        self.assertEqual(transformer.scale_shift_table.dtype, torch.float32)
        self.assertEqual(transformer.blocks[0].scale_shift_table.dtype, torch.float32)
        self.assertEqual(transformer.blocks[0].norm2.weight.dtype, torch.float32)

        MODULE._align_self_forcing_transformer_dtype(transformer)

        self.assertEqual(transformer.condition_embedder.time_embedder[0].weight.dtype, torch.bfloat16)
        self.assertEqual(transformer.condition_embedder.time_embedder[2].weight.dtype, torch.bfloat16)
        self.assertEqual(transformer.scale_shift_table.dtype, torch.bfloat16)
        self.assertEqual(transformer.blocks[0].scale_shift_table.dtype, torch.bfloat16)
        self.assertEqual(transformer.blocks[0].norm2.weight.dtype, torch.bfloat16)
        self.assertEqual(transformer.blocks[0].norm2.bias.dtype, torch.bfloat16)

        transformer = DummyTransformer()
        EXAMPLE_MODULE._align_self_forcing_transformer_dtype(transformer)

        self.assertEqual(transformer.condition_embedder.time_embedder[0].weight.dtype, torch.bfloat16)
        self.assertEqual(transformer.scale_shift_table.dtype, torch.bfloat16)
        self.assertEqual(transformer.blocks[0].scale_shift_table.dtype, torch.bfloat16)
        self.assertEqual(transformer.blocks[0].norm2.weight.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
