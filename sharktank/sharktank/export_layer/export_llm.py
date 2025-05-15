# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.layers.configs import LlamaModelConfig
from sharktank.utils.export_artifacts import ExportArtifacts


def llm_model_config_to_export_cli_args(config: LlamaModelConfig) -> list[str]:
    """Prepare CLI arguments for the export_paged_llm_v1 tool from a model config."""
    properties = config.to_properties()
    config.attention_kernel
    res = [
        f"--attention-kernel={properties['attention_kernel']}",
        f"--use-hf={properties['use_hf']}",
        f"--activation-dtype={properties['activation_dtype']}",
        f"--attention-dtype={properties['attention_dtype']}",
    ]
    if "kv_cache_dtype" in properties:
        res.append(f"--kv-cache-dtype={properties['kv_cache_dtype']}")
    res += [
        f"--tensor-parallelism-size={properties['tensor_parallelism_size']}",
        f"--pipeline-parallelism-size={properties['pipeline_parallelism_size']}",
        f"--block-seq-stride={properties['block_seq_stride']}",
    ]
    return res


def export_paged_llm_v1_from_config(
    config: LlamaModelConfig,
    irpa_path: str,
    batch_size: int,
    iree_hip_target: str,
    iree_hal_target_device: str,
    output_mlir: str,
    use_attention_mask: bool = False,
    output_config: str | None = None,
    skip_decode: bool = True,
):
    properties = config.to_properties()
    kv_cache_dtype = (
        properties["kv_cache_dtype"] if "kv_cache_dtype" in properties else None
    )
    export_artifacts = ExportArtifacts(
        irpa_path=irpa_path,
        batch_size=batch_size,
        iree_hip_target=iree_hip_target,
        attention_kernel=config.attention_kernel,
        tensor_parallelism_size=config.tensor_parallelism_size,
        pipeline_parallelism_size=config.pipeline_parallelism_size,
        block_seq_stride=config.block_seq_stride,
        iree_hal_target_device=iree_hal_target_device,
        use_attention_mask=use_attention_mask,
        use_hf=config.use_hf,
        activation_dtype=properties["activation_dtype"],
        attention_dtype=properties["attention_dtype"],
        kv_cache_dtype=kv_cache_dtype,
        output_mlir=output_mlir,
        output_config=output_config,
    )
    export_artifacts.export_to_mlir()
