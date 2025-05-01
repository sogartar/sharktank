# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from sharktank.types import Theta
from sharktank.layers import *

from sharktank.ops import softmax, topk

__all__ = [
    "MoeBlock",
]


class SparseMoeBlock(ThetaLayer):
    """
    This implementation considers MoE operations as block-sparse
    operations to support imbalanced token assignments to experts.
    This enables the MoE to operate at a faster rate and in full capacity without any dropped tokens
    (or reduced performance).
    """

    def __init__(
        self,
        theta: Theta,
        expert_count: int,
        expert_used_count: int,
        rms_epsilon: float,
    ):
        super().__init__(theta)

        # Add router gate
        self.add_module("ffn_gate_inp", LinearLayer(theta("ffn_gate_inp")))

        # Add FFN norm
        self.add_module(
            "ffn_norm", RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)
        )

        # Add FFN output norm
        self.add_module(
            "layer_output_norm",
            RMSNormLayer(theta("layer_output_norm"), epsilon=rms_epsilon),
        )

        # Add expert_count x FFN
        self.experts = nn.ModuleList(
            [SparseFFNMOE(theta, expert_idx=i) for i in range(expert_count)]
        )

        self.expert_count = expert_count
        self.expert_used_count = expert_used_count

    def forward(
        self,
        h: torch.Tensor,
    ):
        ffn_input = self.ffn_norm(h)
        batch_size, sequence_length, feature_dim = ffn_input.shape
        ffn_input = ffn_input.view(-1, feature_dim)

        # For each token, the router calculates the router weights for all experts
        # router_logits: (batch_size * sequence_length, expert_count)
        router_logits = self.ffn_gate_inp(ffn_input)
        router_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # Select top k experts from router weights
        router_weights, top_k_experts = torch.topk(
            router_weights, self.expert_used_count, dim=-1
        )
        router_weights /= router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(ffn_input.dtype)

        moe_output = torch.zeros(
            (batch_size * sequence_length, feature_dim), dtype=ffn_input.dtype
        )

        # Create an expert mask by one hot encoding the selected top k experts
        # used to index which expert is to be invoked for each token
        # expert_mask: (expert_count, expert_used_count, sequence_length)
        expert_mask = F.one_hot(top_k_experts, num_classes=self.expert_count).permute(
            2, 1, 0
        )

        # Iterate over all experts in the model
        for expert_idx in range(self.expert_count):
            expert_layer = self.experts[expert_idx]
            top_k_expert_idx, token_idx = torch.where(expert_mask[expert_idx])

            # Given the hidden states, index the tokens assigned to this expert
            # and calculate the current expert's hidden state and weigh the
            # output expert hidden states by the router weights, based on the
            # appropriate tokens
            current_expert_tokens = ffn_input[None, token_idx]

            current_expert = (
                expert_layer(current_expert_tokens)
                * router_weights[token_idx, top_k_expert_idx, None]
            )

            current_expert = current_expert.reshape(-1, feature_dim)

            moe_output.index_add_(0, token_idx, current_expert.to(ffn_input.dtype))
        moe_output = moe_output.reshape(batch_size, sequence_length, feature_dim)

        moe_output = self.layer_output_norm(moe_output)
        return h + moe_output


class MoeBlock(ThetaLayer):
    """
    This implementation considers MoE operations as block-sparse
    operations to support imbalanced token assignments to experts.
    This enables the MoE to operate at a faster rate and in full capacity without any dropped tokens
    (or reduced performance).
    """

    def __init__(
        self,
        theta: Theta,
        expert_used_count: int,
        rms_epsilon: float,
        moe_activation=torch.nn.functional.silu,
        *,
        experts_ffn_moe_block: PreGatherFFNMOE | DenseFFNMOE | str,
        score_experts=softmax,
        normalize_experts=True,
        add_residual=True,
        route_scale: Optional[float] = 1.0,
    ):
        super().__init__(theta)
        self.expert_used_count = expert_used_count
        self.score_experts = score_experts
        self.normalize_experts = normalize_experts
        self.add_residual = add_residual

        # Add router gate
        self.add_module("ffn_gate_inp", LinearLayer(theta("ffn_gate_inp")))

        self.ffn_norm = torch.nn.Identity()
        self.layer_output_norm = torch.nn.Identity()
        self.shared_experts = None
        self.route_scale = route_scale if route_scale > 1 else None

        # Add FFN norm
        if "ffn_norm" in theta:
            self.ffn_norm = RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)

        # Add expert_count x FFN
        if isinstance(experts_ffn_moe_block, str):
            if experts_ffn_moe_block == "PreGatherFFNMOE":
                self.experts = PreGatherFFNMOE(theta, activation=moe_activation)
            elif experts_ffn_moe_block == "DenseFFNMOE":
                self.experts = DenseFFNMOE(theta, activation_fn=moe_activation)
            else:
                raise ValueError(
                    f'Unknown experts_ffn_moe_block "{experts_ffn_moe_block}"'
                )
        else:
            self.experts = experts_ffn_moe_block

        if "shared_experts" in theta:
            self.shared_experts = FFN(theta("shared_experts"))

        # Add optional FFN output norm layer
        if theta.optional_tensor("layer_output_norm") is not None:
            self.layer_output_norm = RMSNormLayer(
                theta("layer_output_norm"), epsilon=rms_epsilon
            )

    def forward(
        self,
        h: torch.Tensor,
    ):
        ffn_input = self.ffn_norm(h)
        batch_size, sequence_length, feature_dim = ffn_input.shape
        ffn_input = ffn_input.view(-1, feature_dim)

        # For each token, the router calculates the router weights for all experts
        # router_logits: (batch_size * sequence_length, expert_count)
        router_logits = self.ffn_gate_inp(ffn_input)
        router_weights = self.score_experts(router_logits.to(torch.float))

        # Select top k experts from router weights
        expert_gate, top_k_experts = topk(
            router_weights, self.expert_used_count, dim=-1
        )

        if self.normalize_experts:
            expert_gate /= expert_gate.sum(dim=-1, keepdim=True)

        expert_gate = expert_gate.to(ffn_input.dtype)

        if self.route_scale is not None:
            expert_gate = expert_gate * self.route_scale

        moe_output = self.experts(ffn_input, top_k_experts, expert_gate)

        if self.shared_experts:
            moe_output = moe_output + self.shared_experts(ffn_input)

        moe_output = moe_output.reshape(batch_size, sequence_length, feature_dim)

        moe_output = self.layer_output_norm(moe_output)
        if self.add_residual:
            moe_output = h + moe_output

        return moe_output
