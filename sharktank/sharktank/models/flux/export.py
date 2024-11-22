# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import os

from ...utils.hf import convert_hf_to_gguf

__all__ = [
    "convert_flux_hf_to_gguf",
]


def convert_flux_hf_to_gguf(
    hf_model_path: str, text_encoder_2_output_path: Optional[str] = None
):
    if text_encoder_2_output_path is not None:
        convert_hf_to_gguf(
            hf_model_path=os.path.join(hf_model_path, "text_encoder_2"),
            hf_tokenizer_path=os.path.join(hf_model_path, "tokenizer_2"),
            output_path=text_encoder_2_output_path,
        )
