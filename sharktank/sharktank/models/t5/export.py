# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from typing import Optional, Union
from pathlib import Path
from os import PathLike
import torch
from copy import copy

from .t5 import T5Config, T5Encoder
from ...types import Dataset, Theta, torch_module_to_theta
from ...transforms.dataset import set_float_dtype
from iree.turbine.aot import FxProgramsBuilder, export
import transformers

__all__ = [
    "export_encoder_mlir",
    "export_encoder_iree_parameters",
    "export_encoder_to_iree",
    "hugging_face_encoder_to_theta",
]


def hugging_face_encoder_to_theta(model: transformers.CLIPTextModel) -> Theta:
    return torch_module_to_theta(model)


def hugging_face_encoder_to_dataset(
    model: transformers.T5EncoderModel,
) -> Dataset:
    config = T5Config.from_hugging_face_config(model.config)
    properties = config.to_properties()
    theta = hugging_face_encoder_to_theta(model)
    theta.rename_tensors_to_paths()
    return Dataset(properties, theta)


def encoder_to_dataset(model: T5Encoder) -> Dataset:
    return Dataset(properties=model.config.to_properties(), root_theta=model.theta)


def encoder_model_to_dataset(model: T5Encoder) -> Dataset:
    return Dataset(properties=model.config.to_properties(), root_theta=model.theta)


def export_encoder_mlir(
    model: Union[T5Encoder, Path, str],
    batch_sizes: list[int],
    mlir_output_path: str,
):
    """
    Args:
      model: either the torch module or path to GGUF/IRPA.
    """
    if isinstance(model, (Path, str)):
        dataset = Dataset.load(model)
        config = T5Config.from_gguf_properties(
            dataset.properties,
            # TODO: add this property to our HuggingFace-to-GGUF conversion script.
            # We currently use llama.cpp's converter and it can not make a distinction
            # between T5 V1 and V1.1.
            # V1 uses ReLU and V1.1 uses gated GeLU.
            feed_forward_proj="gated-gelu",
        )
        model = T5Encoder(theta=dataset.root_theta, config=config)

    fxb = FxProgramsBuilder(model)

    for batch_size in batch_sizes:
        sample_inputs = model.sample_inputs(batch_size)

        context_length_dim_idx = 1
        assert (
            sample_inputs["input_ids"].shape[context_length_dim_idx]
            % config.context_length_padding_block_size
            == 0
        )
        context_length_block_dim_max = (
            sample_inputs["input_ids"].shape[context_length_dim_idx]
            // config.context_length_padding_block_size
        )
        context_length_block_dim = torch.export.Dim(
            "block", max=context_length_block_dim_max
        )
        context_length_dim = (
            config.context_length_padding_block_size * context_length_block_dim
        )
        dynamic_shapes = {"input_ids": {context_length_dim_idx: context_length_dim}}

        @fxb.export_program(
            name=f"forward_bs{batch_size}",
            args=tuple(sample_inputs.values()),
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
        def _(
            model,
            input_ids,
        ):
            return model(input_ids)

    output = export(fxb, import_symbolic_shape_expressions=True)
    output.save_mlir(mlir_output_path)


def export_encoder_iree_parameters(model: T5Config, output_path: PathLike):
    dataset = encoder_model_to_dataset(model)
    dataset.save(output_path)

def export_encoder_to_iree(
    model: T5Encoder,
    batch_sizes: list[int],
    mlir_output_path: PathLike,
    parameters_output_path: PathLike,
):
    export_encoder_iree_parameters(model, parameters_output_path)
    export_encoder_mlir(
        model=parameters_output_path,
        batch_sizes=batch_sizes,
        mlir_output_path=mlir_output_path,
    )
