# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch
from iree.turbine.aot import *
from sharktank.models.llama.testing import make_moe_block_theta, make_rand_torch
from sharktank.layers.mixture_of_experts_block import MoeBlock


class MoeBlockTest(unittest.TestCase):
    def testExport(self):
        model = MoeBlock(
            theta=make_moe_block_theta()("blk.0"),
            expert_used_count=2,
            rms_epsilon=1e-5,
        )
        fxb = FxProgramsBuilder(model)
        input = make_rand_torch((2, 32, 6144))

        @fxb.export_program(name="moe_block", args=(input,), strict=False)
        def _(model, input: torch.Tensor) -> torch.Tensor:
            return model(input)

    def testParityOfExpertPreGatherFfnAndDenseFfn(self):
        from sharktank.layers.testing import make_random_moe_block_theta
        from sharktank.layers import MoeBlock

        dtype = torch.float32
        feature_dim = 7
        expert_hidden_dim = 3
        num_experts = 5
        expert_used_count = 2
        num_shared_experts = 11
        shared_expert_hidden_dim = 13
        batch_size = 17
        sequence_length = 19
        rms_epsilon = 0.01
        moe_activation = torch.nn.functional.silu
        score_experts = torch.nn.functional.sigmoid
        normalize_experts = True
        add_residual = False
        route_scale = 1.5

        theta = make_random_moe_block_theta(
            in_dim=feature_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            with_ffn_norm=True,
            num_shared_experts=num_shared_experts,
            shared_expert_hidden_dim=shared_expert_hidden_dim,
            with_layer_output_norm=True,
            dtype=dtype,
        )

        moe_with_pre_gather_ffn = MoeBlock(
            theta=theta,
            expert_used_count=expert_used_count,
            rms_epsilon=rms_epsilon,
            moe_activation=moe_activation,
            experts_ffn_moe_block="PreGatherFFNMOE",
            score_experts=score_experts,
            normalize_experts=normalize_experts,
            add_residual=add_residual,
            route_scale=route_scale,
        )
        moe_with_dense_ffn = MoeBlock(
            theta=theta,
            expert_used_count=expert_used_count,
            rms_epsilon=rms_epsilon,
            moe_activation=moe_activation,
            experts_ffn_moe_block="DenseFFNMOE",
            score_experts=score_experts,
            normalize_experts=normalize_experts,
            add_residual=add_residual,
            route_scale=route_scale,
        )

        input = (
            torch.rand([batch_size, sequence_length, feature_dim], dtype=dtype) - 0.5
        )
        res_pre_gather = moe_with_pre_gather_ffn(input)
        res_dense = moe_with_dense_ffn(input)
        torch.testing.assert_close(res_pre_gather, res_dense)


if __name__ == "__main__":
    unittest.main()
