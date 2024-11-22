# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import os
from tempfile import TemporaryDirectory
import subprocess
import sys
import shutil

__all__ = [
    "convert_hf_to_gguf",
]


def _symlink_all_direct_subitems(source_directory: str, target_directory: str):
    for file_name in os.listdir(source_directory):
        os.symlink(
            os.path.join(source_directory, file_name),
            os.path.join(target_directory, file_name),
        )


def _merge_models(model_paths: list[str], output_path: str):
    for model_path in model_paths:
        _symlink_all_direct_subitems(model_path, output_path)


def convert_hf_to_gguf(
    hf_model_path: str, output_path: str, hf_tokenizer_path: Optional[str] = None
):
    """Conver a HuggingFace model to GGUF.
    This function assumes that llama.cpp's convert_hf_to_gguf.py is in your path.
    See https://github.com/ggerganov/llama.cpp"""
    # TODO add a way of providing llama.cpp's convert_hf_to_gguf.py
    if hf_tokenizer_path is not None:
        # convert_hf_to_gguf.py expects the tokenizer in the same directory as the model.
        # Make a temp directory with symlinks populated to both.
        with TemporaryDirectory(
            prefix=f"{os.path.basename(output_path)}_", dir=os.path.dirname(output_path)
        ) as tmp_dir:
            _merge_models([hf_model_path, hf_tokenizer_path], output_path=tmp_dir)
            convert_hf_to_gguf(tmp_dir, output_path)
            return

    subprocess.check_call(
        [
            sys.executable,
            shutil.which("convert_hf_to_gguf.py"),
            f"--outfile={output_path}",
            hf_model_path,
        ]
    )
