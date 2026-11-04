# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.metadata

from packaging import version
from transformers import PreTrainedModel

from arctic_training.model.hf_factory import HFModelFactory


class LigerModelFactory(HFModelFactory):
    name = "liger"

    def create_model(self, model_config) -> PreTrainedModel:
        try:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
        except ImportError:
            raise ImportError(
                "You need to install the liger-kernel package to use LigerKernel models: `pip install liger-kernel`"
            )
        liger_version_min = "0.6.1"  # int64 indexing
        liger_version_have = importlib.metadata.version("liger_kernel")
        if version.parse(liger_version_have) < version.parse(liger_version_min):
            raise ValueError(
                f"liger-kernel>={liger_version_min} is required, but you have liger-kernel=={liger_version_have}"
            )

        # Restrict liger to fused linear cross-entropy only; leave rope, RMSNorm and SwiGLU
        # to HF. With those kernels on, native_activation_checkpoint_cpu_offload regresses
        # badly: +70 GiB peak at seq 32768 on Qwen3-8B (SwiGLU ~54, RMSNorm ~29, rope 0),
        # versus no penalty with FLCE alone.
        #
        # Mechanism (measured 2026-10-16, autorun/diag_offload*.py + diag_memhist.py):
        # with liger the checkpointed layers' FORWARD activations are never released --
        # 84.96 GiB live at end of forward vs 29.37 GiB without, 54 GiB of it SwiGLU's
        # gate/up projections (linear.py:134 <- swiglu.py:35 = 36 layers x 2 x 768 MiB).
        # The checkpoint pack hook DOES intercept those tensors (108 x (32768,12288) =
        # 81 GiB packed) and gc.get_referrers() finds no Python holder, so the retaining
        # reference is C-level inside autograd/liger and is not fixable from the DeepSpeed
        # fork. autorun/bench_qwen3_offload.py pins this same FLCE-only set.
        # NB: an earlier "leaf-set accounting" explanation of this was WRONG -- that block
        # frees no memory and is not even installed by default (SYNCHRONIZE/PROFILE_TIME
        # both False).
        #
        # This also removes the MLP monkey-patch race with tiled_mlp_compute that the
        # previous `swiglu = not tiled_mlp_compute` line existed to avoid -- SwiGLU is now
        # always left to HF (or to tiled_mlp_compute), so the two can never contend.
        try:

            if self.using_random_model:
                # skip the weight loading for a faster startup if we are in a random model configuration mode
                return AutoLigerKernelForCausalLM.from_config(
                    model_config,
                    attn_implementation=self.config.attn_implementation,
                    dtype=self.config.dtype.value,
                    rope=False,
                    rms_norm=False,
                    swiglu=False,
                    cross_entropy=False,
                    fused_linear_cross_entropy=True,
                )
            else:
                return AutoLigerKernelForCausalLM.from_pretrained(
                    self.config.name_or_path,
                    config=model_config,
                    attn_implementation=self.config.attn_implementation,
                    dtype=self.config.dtype.value,
                    rope=False,
                    rms_norm=False,
                    swiglu=False,
                    cross_entropy=False,
                    fused_linear_cross_entropy=True,
                )
        except KeyError as e:
            raise ValueError(
                f"It appears that liger-kernel=={liger_version_have} doesn't support the architecture of"
                f" {self.config.name_or_path}, perhaps try the latest liger-kernel version. It can't find {e}."
            )
        except Exception:
            raise
