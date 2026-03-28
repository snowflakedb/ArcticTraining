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

from typing import Optional

import deepspeed
import torch
import wandb
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPDataLoaderAdapter
from devtools import debug
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from wandb.sdk.wandb_run import Run as WandbRun

from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.trainer import TrainerConfig
from arctic_training.data.utils import OverfitOneBatchDataLoader
from arctic_training.debug.utils import pr0
from arctic_training.debug.utils import see_memory_usage
from arctic_training.logging import logger
from arctic_training.metrics import Metrics
from arctic_training.model.tiled_compute import enable_tiled_mlp_compute
from arctic_training.trainer.trainer import Trainer


class MoETrainer(Trainer):
    """Trainer subclass that adds Arctic MoE support.

    Overrides ``__init__`` and ``checkpoint`` to insert MoE model detection,
    parameter remapping, DeepSpeed profiler instrumentation, and checkpoint
    export back to the original MoE layout.
    """

    def __init__(self, config: TrainerConfig, mode: str = "train") -> None:
        logger.info(f"Initializing Trainer with config:\n{debug.format(config)}")
        self.config = config
        self.epoch_idx = 0
        self.train_batch_idx = 0
        self.global_step = 0
        self.global_step_this_run = 0
        self.global_step_at_start_this_run = 0
        self.early_stop = False
        self.early_stop_reason = ""
        self.world_size = config.world_size
        self.global_rank = config.global_rank
        self.epoch_finished = False
        self.training_finished = False
        self.wandb_experiment: Optional[WandbRun] = None
        self.is_resume = False  # Track if we resumed from ckpt
        self.wandb_run_id = None

        self._set_seeds(self.config.seed)

        if self.config.mem_profiler == "e2e":
            torch.cuda.memory._record_memory_history(max_entries=self.config.mem_profiler_max_entries)

        tokenizer_factory = self.config.tokenizer.factory(self)
        self.tokenizer = tokenizer_factory()

        data_factory = self.config.data.factory(self)
        self.train_dataloader, self.eval_dataloader = data_factory()
        if mode == "process-data":
            return

        if self.config.overfit_first_batch:
            self.train_dataloader = OverfitOneBatchDataLoader(self.train_dataloader)

        # checkpointing and resume
        self.checkpoint_engines = [engine(self) for engine in self.config.checkpoint_engines]
        for engine in self.checkpoint_engines:
            # currently only deepspeed engine supports resume from intermediate checkpoint
            if engine.name == "deepspeed" and engine.config.auto_resume and engine.latest_checkpoint_exists:
                self.is_resume = True

        print(f"IS RESUME={self.is_resume}")

        # XXX: We can abstract this section further with AT-specific wrapper, but
        # UlyssesSPAttentionHF should not have any AT-specific objects / assumptions
        mpu = UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=self.config.model.name_or_path,
            core_attn_implementation=self.config.model.attn_implementation,
            sequence_parallel_size=self.config.sequence_parallel_size,
            micro_batch_size=self.config.micro_batch_size,
            seq_length=self.config.data.max_length,
            seq_length_is_variable=True,
        )

        # Important: this is most likely not beneficial under seqlen=64k
        if self.config.activation_checkpoint_cpu_offload:
            # activation_checkpointing_cpu_offload becomes very benefitial at very long seqlen
            # e.g., llama 8b at 800k (100k effective per gpu) will save 24GB per gpu:
            # ((100_000*4096)*2*32/2**30), but for short sequences the offload will just slow things
            # down,
            #
            # XXX: could parameterize or run a few lengths to see at which threshold it becomes
            # beneficial - a user might still want this on even at shorter seqlen if they don't
            # mind slower performance. discussing adding this functionality to pytorch core
            # (https://pytorch.slack.com/archives/C3PDTEV8E/p1745274102600729)
            from arctic_training.monkey_patches import monkey_patch_checkpoint_function_with_cpu_offload

            monkey_patch_checkpoint_function_with_cpu_offload()

        # MLP tiling - has to happen before model is instantiated
        if self.config.tiled_mlp_compute:
            enable_tiled_mlp_compute(self.config.model.name_or_path)

        dschf = HfDeepSpeedConfig(self.config.deepspeed)  # noqa: F841
        model_factory = self.config.model.factory(self)
        self.model = model_factory()

        self.count_model_params_in_original_model()

        # prevent causal mask from being created in HF Transformers - it's a huge `[bs, seqlen, seqlen]` tensor
        # XXX: This should also benefit a single gpu use case when SDPA is used - so perhaps remove the SP>1 check?
        if self.config.sequence_parallel_size > 1 and self.config.model.attn_implementation not in [
            "flash_attention_2",
            "flash_attention_3",
        ]:
            import transformers.masking_utils

            transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", lambda *args, **kwargs: None)

        # ---- Arctic MoE setup (must happen before optimizer creation) ----
        from arctic_training.model.moe.utils import detect_if_moe_model
        from arctic_training.model.moe.utils import remap_orig_moe_mlp_params_to_arctic_moe

        if self.config.arctic_moe == "auto":
            self.use_arctic_moe = detect_if_moe_model(self.model)
        else:
            self.use_arctic_moe = self.config.arctic_moe

        if self.use_arctic_moe:
            pr0("Activating ArcticMoE", force=True)
            import deepspeed.comm as dist

            if not dist.is_initialized():
                dist.init_distributed(dist_backend="nccl", dist_init_required=True)

            from arctic_training.model.moe.utils import monkey_patch_ds_moe

            monkey_patch_ds_moe()

            # deepspeed.runtime.engine.DeepSpeedEngine.print_forward_breakdown = print_forward_breakdown
            # DeepspeedMoE is only integrated with ZeRO-2
            zero_stage = self.config.deepspeed.get("zero_optimization", {}).get("stage", 0)
            if zero_stage != 2:
                raise ValueError(
                    "at the moment Deepspeed supports only ZeRO stage 2 with MoE, but the configuration asks for ZeRO"
                    f" stage={zero_stage}"
                )

            from deepspeed.utils import groups

            # this config comes from use_data_before_expert_parallelism ds config which defaults to False
            # engine._config.use_data_before_expert_parallel_)
            # but we don't have the engine yet to get the ds config values - perhaps could extract this via AT-config?
            use_data_before_expert_parallel_ = False
            # the ep group has to be created before remap_orig_moe_mlp_params_to_arctic_moe as ep rank info is needed to remap pre-trained experts
            groups._create_expert_data_and_model_parallel(
                self.config.expert_parallel_size,
                mpu=None,
                use_data_before_expert_parallel_=use_data_before_expert_parallel_,
            )

            # self.groups = ParallelGroups(expert_parallel_size=self.config.expert_parallel_size)

            # we sort out if we are in resume mode much later, by actually trying to load the model, but that's too late so we are going to rely on testing if the latest checkpoint exists instead
            # early_is_resume = False
            # for engine in self.checkpoint_engines:
            #     # currently only deepspeed engine supports resume from intermediate checkpoint
            #     if engine.name == "deepspeed" and engine.config.auto_resume and engine.latest_checkpoint_exists:
            #         early_is_resume = True
            remap_orig_moe_mlp_params_to_arctic_moe(
                self.model, ep_size=self.config.expert_parallel_size, is_resume=self.is_resume
            )
            # self.groups)
            # XXX: check we can remap back
            # from arctic_training.model.moe.utils import remap_arctic_moe_params_to_orig_moe_mlp
            # remap_arctic_moe_params_to_orig_moe_mlp(self.model)

        see_memory_usage("after moe remap", force=False)

        optimizer_factory = self.config.optimizer.factory(self)
        self.optimizer = optimizer_factory()

        scheduler_factory = self.config.scheduler.factory(self)
        self.scheduler = scheduler_factory()

        see_memory_usage("before deepspeed.initialize", force=False)

        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            args=self.config,
            lr_scheduler=self.scheduler,
            config=self.config.deepspeed,
            mpu=mpu,
        )
        see_memory_usage("after deepspeed.initialize", force=False)

        # ---- Arctic MoE post-DS instrumentation ----
        if self.use_arctic_moe:
            from arctic_training.model.moe.moe import ArcticMoE

            for module in self.model_unwrapped.modules():
                if isinstance(module, ArcticMoE):
                    # self.model.gate_modules.append(module)
                    if self.model.wall_clock_breakdown():
                        module.enable_wall_clock_breakdown()

        self.ds_wall_clock_available = hasattr(self.model, "get_wall_clock_timers")

        if self.config.sequence_parallel_size > 1:
            # deepspeed.initialize needs to run first
            from deepspeed.utils import groups

            # set SP-trainer attributes to be used later
            self.sp_group = groups._get_sequence_parallel_group()
            self.sp_world_size = groups._get_sequence_parallel_world_size()
            self.sp_rank = groups._get_sequence_parallel_rank()

            # wrap the DL with Ulysses one
            self.train_dataloader = UlyssesSPDataLoaderAdapter(
                self.train_dataloader,
                sp_rank=self.sp_rank,
                sp_group=self.sp_group,
                sp_world_size=self.sp_world_size,
                device=self.device,
            )

            if self.eval_dataloader is not None:
                self.eval_dataloader = UlyssesSPDataLoaderAdapter(
                    self.eval_dataloader,
                    sp_rank=self.sp_rank,
                    sp_group=self.sp_group,
                    sp_world_size=self.sp_world_size,
                    device=self.device,
                )

        for engine in self.checkpoint_engines:
            if engine.config.auto_resume:
                engine.load(self.model)

        self.metrics = Metrics(self)

        if self.global_rank == 0 and self.config.wandb.enable:

            # in order for resume to continue the same wandb run we need to re-use a run_id from the previous run
            if self.wandb_run_id is None:
                self.wandb_run_id = wandb.util.generate_id()

            # Note: wandb.init() is not type annotated so we need to use type: ignore
            self.wandb_experiment = wandb.init(  # type: ignore
                id=self.wandb_run_id,
                entity=self.config.wandb.entity,
                project=self.config.wandb.project,
                name=self.config.wandb.name,
                config=self.config.model_dump(),
                # do not put `wandb` in the root of the repo as it conflicts with wandb package
                dir=f"{self.config.logger.output_dir}/wandb",
            )

    @callback_wrapper("checkpoint")
    def checkpoint(self) -> None:

        pr0(f"{self.global_step_this_run=}")
        if self.global_step_this_run == 0:
            logger.info("No steps were run this run, not saving the checkpoint")
            return

        for engine in self.checkpoint_engines:
            if engine.do_checkpoint:

                if engine.name == "huggingface" and self.use_arctic_moe:
                    if self.training_finished:
                        # export to the original moe mlp format/layout - this is slow but it's the end of the training so it's fine.
                        from arctic_training.model.moe.utils import remap_arctic_moe_to_orig_moe_mlp_params

                        logger.info("Exporting to the original MoE format before saving the checkpoint")
                        remap_arctic_moe_to_orig_moe_mlp_params(self.model)
                    else:
                        raise ValueError(
                            "Currently supporting saving to HF checkpoint for AMoE models only when the training is"
                            " finished, because conversion will be very slow. For interim checkpoints use `deepspeed`"
                            " type of the checkpoint as it'd be much faster to save to and resume from. "
                        )

                logger.info(f"Saving Checkpoint at global step: {self.global_step}.")
                engine.save(self.model)

    def print_model_parameters_header(self):
        """Extends base header with EP-specific parameter counts."""
        if torch.distributed.get_rank() != 0:
            return

        orig_model_params = self.original_hf_model_params
        curr_model_params = self.count_model_parameters()

        world_size = self.world_size
        gas = self.config.gradient_accumulation_steps
        mbs = self.config.micro_batch_size
        gbs = mbs * gas * world_size

        header = f"""
-------------------------------------
Original model: {self.config.model.name_or_path}
    - Total params    : {orig_model_params["total"]:,} ({orig_model_params["total"]/1e9:0.2f}B)
    - Trainable params: {orig_model_params["trainable"]:,} ({orig_model_params["trainable"]/1e9:.2f}B)
"""

        if self.config.expert_parallel_size > 1:
            header += f"""
Rank 0 model with EP={self.config.expert_parallel_size}:
    - Total params    : {curr_model_params["total"]:,} ({curr_model_params["total"]/1e9:0.2f}B)
    - Trainable params: {curr_model_params["trainable"]:,} ({curr_model_params["trainable"]/1e9:.2f}B)
"""

        header += f"""
Parallelism:
    - EP: {self.config.expert_parallel_size}
    - SP: {self.config.sequence_parallel_size}
    - DP: {world_size}
    """

        header += f"""
Maximum number of optimizer steps: {self.config.exit_iteration}
Maximum number of epochs: {self.config.epochs}
Number of gradient accumulation steps: {gas}
Number of processes: {world_size}
Batch sizes:
    - Micro  batch size: {mbs}
    - Global batch size: {gbs}
-------------------------------------
"""

        print(header)
