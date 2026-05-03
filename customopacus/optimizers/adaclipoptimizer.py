# Copyright (c) Meta Platforms, Inc. and affiliates.
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

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
from torch.optim import Optimizer

from .optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _generate_noise,
    _mark_as_processed,
)


logger = logging.getLogger(__name__)


class AdaClipDPOptimizer(DPOptimizer):
    """
    :class:`~customopacus.optimizers.optimizer.DPOptimizer` that implements
    adaptive clipping strategy
    https://arxiv.org/pdf/1905.03871.pdf
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        Vk,
        noise_multiplier: float,
        target_unclipped_quantile: float = 0.5,
        clipbound_learning_rate: float = 0.2,
        max_clipbound: float = 3,
        min_clipbound: float = 0.1,
        unclipped_num_std: float = 1.0,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            optimizer,
            Vk=Vk,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )
        assert (
            max_clipbound > min_clipbound
        ), "max_clipbound must be larger than min_clipbound."
        self.target_unclipped_quantile = target_unclipped_quantile
        self.clipbound_learning_rate = clipbound_learning_rate
        self.max_clipbound = max_clipbound
        self.min_clipbound = min_clipbound
        self.unclipped_num_std = unclipped_num_std
        # Theorem 1. in  https://arxiv.org/pdf/1905.03871.pdf
        self.noise_multiplier = (
            self.noise_multiplier ** (-2) - (2 * unclipped_num_std) ** (-2)
        ) ** (-1 / 2)
        self.sample_size = 0
        self.unclipped_num = 0

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients, self.sample_size and self.unclipped_num
        """
        super().zero_grad(set_to_none)

        self.sample_size = 0
        self.unclipped_num = 0

    def clip_and_accumulate(self):
        B = self.expected_batch_size
        K = len(self.grad_samples[0]) // B if B > 0 else 0

        has_projection = self.Vk is not None
        needs_averaging = K > 1

        per_sample_clip_factor = None
        processed_grad_samples = []

        # 第一阶段：计算梯度、投影（可选）、平均（可选）

        if len(self.grad_samples[0]) == 0:
            per_sample_clip_factor = torch.zeros((0,), device=self.grad_samples[0].device)
            processed_grad_samples = []

        else:
            # 1. 展平所有梯度以便处理
            grads_flat = [g.reshape(len(g), -1) for g in self.grad_samples]
            all_grads = torch.cat(grads_flat, dim=1)

            current_batch_size = len(all_grads)  # 实际是 B * K

            # 2. 投影步骤 (如果 Vk 存在)
            if has_projection:
                processed_all = torch.matmul(torch.matmul(all_grads, self.Vk), self.Vk.T)
            else:
                processed_all = all_grads

            # 3. 平均步骤 (如果 K > 1)
            if needs_averaging:
                reshaped_grads = processed_all.view(B, K, *processed_all.shape[1:])
                # final_grads = reshaped_grads.mean(dim=1)
                final_grads = reshaped_grads.sum(dim=1)
            else:
                final_grads = processed_all

            # 4. 计算裁剪因子
            per_sample_norms = torch.norm(final_grads, dim=1)
            per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)

            # 5. 切分回各个参数的形状
            start_idx = 0
            for grad in self.grad_samples:
                param_flat_dim = grad.reshape(len(grad), -1).shape[1]
                end_idx = start_idx + param_flat_dim

                split_grad = final_grads[:, start_idx:end_idx].view(-1, *grad.shape[1:])
                processed_grad_samples.append(split_grad)

                start_idx = end_idx

        # 第二阶段：统一应用裁剪并累加
        if per_sample_clip_factor is None:
            per_sample_clip_factor = torch.zeros((0,), device=self.grad_samples[0].device)

        for p, grad_sample in zip(self.params, processed_grad_samples):
            _check_processed_flag(p.grad_sample)

            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

        # 自适应裁剪
        self.unclipped_num += (
            len(per_sample_clip_factor) - (per_sample_clip_factor < 1).sum()
        )

    def add_noise(self):
        super().add_noise()

        unclipped_num_noise = _generate_noise(
            std=self.unclipped_num_std,
            reference=self.unclipped_num,
            generator=self.generator,
        )

        self.unclipped_num = float(self.unclipped_num)
        self.unclipped_num += unclipped_num_noise

    def update_max_grad_norm(self):
        """
        Update clipping bound based on unclipped fraction
        """
        unclipped_frac = self.unclipped_num / self.sample_size
        self.max_grad_norm *= torch.exp(
            -self.clipbound_learning_rate
            * (unclipped_frac - self.target_unclipped_quantile)
        )
        if self.max_grad_norm > self.max_clipbound:
            self.max_grad_norm = self.max_clipbound
        elif self.max_grad_norm < self.min_clipbound:
            self.max_grad_norm = self.min_clipbound

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        pre_step_full = super().pre_step()
        if pre_step_full:
            self.update_max_grad_norm()
        return pre_step_full
