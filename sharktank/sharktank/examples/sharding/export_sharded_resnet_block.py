# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import sys

import torch


from shark_turbine import aot
from ...models.punet.testing import make_resnet_block_2d_theta
from ...utils import cli
from sharktank.models.punet.layers import ResnetBlock2D
from sharktank.models.punet.sharding import ResnetBlock2DSplitOutputChannelsSharding
from sharktank import ops
from sharktank.types import *
import iree.runtime
import argparse

vm_context: iree.runtime.VmContext = None


def run_iree_module(
    sharded_input_image: ShardedTensor,
    sharded_input_time_emb: ShardedTensor,
    module_path: str,
    parameters_path: str,
) -> ShardedTensor:
    # system_context = iree.runtime.SystemContext(iree.runtime.Config("rocm"))
    hal_driver = iree.runtime.get_driver("hip")
    vm_instance = iree.runtime.VmInstance()
    available_devices = hal_driver.query_available_devices()
    assert len(available_devices) > 2
    devices = [
        hal_driver.create_device(available_devices[0]),
        hal_driver.create_device(available_devices[1]),
    ]
    hal_module = iree.runtime.create_hal_module(instance=vm_instance, devices=devices)
    params_path = Path(parameters_path)
    parameter_index = iree.runtime.ParameterIndex()
    parameter_index.load(
        file_path=str(Path(params_path).with_suffix(f".rank{0}{params_path.suffix}"))
    )
    parameter_index.load(
        file_path=str(Path(params_path).with_suffix(f".rank{1}{params_path.suffix}"))
    )
    parameter_provider = parameter_index.create_provider(scope="model")
    parameters_module = iree.runtime.create_io_parameters_module(
        vm_instance, parameter_provider
    )

    vm_module = iree.runtime.VmModule.mmap(vm_instance, str(module_path))

    # The context needs to be destroied after the buffers, although
    # it is not associate with them on the API level.
    global vm_context
    vm_context = iree.runtime.VmContext(
        instance=vm_instance, modules=(hal_module, parameters_module, vm_module)
    )
    print(f"VM context created.")
    module_input_args = [
        iree.runtime.asdevicearray(
            devices[0], sharded_input_image.shards[0].as_torch().to("cpu").numpy()
        ),
        iree.runtime.asdevicearray(
            devices[1], sharded_input_image.shards[1].as_torch().to("cpu").numpy()
        ),
        iree.runtime.asdevicearray(
            devices[0], sharded_input_time_emb.shards[0].as_torch().to("cpu").numpy()
        ),
        iree.runtime.asdevicearray(
            devices[1], sharded_input_time_emb.shards[1].as_torch().to("cpu").numpy()
        ),
    ]
    print(f"args copied to devices.")
    vm_function = vm_module.lookup_function("main")
    print(f"main found.")
    invoker = iree.runtime.FunctionInvoker(
        vm_context=vm_context,
        device=devices[0],
        vm_function=vm_function,
    )
    print(f"Invoking main.")
    results = invoker(*module_input_args)
    print(f"Invoking main done.")
    shards = [torch.tensor(tensor.to_host()) for tensor in results]
    return SplitPrimitiveTensor(ts=shards, shard_dim=1)


def main(argv):
    parser = cli.create_parser(description="Export sharded Resent block.")
    parser.add_argument(
        "--mlir", type=Path, required=True, help="Path to exported MLIR program."
    )
    parser.add_argument(
        "--module", type=Path, required=True, help="Path to exported IREE module."
    )
    parser.add_argument(
        "--parameters", type=Path, required=True, help="Exported model parameters."
    )
    args = cli.parse(parser, args=argv)

    torch.set_default_dtype(torch.float32)
    batches = 2
    in_channels = 6
    out_channels = [12, 8]
    height = 11
    width = 13
    kernel_height = 5
    kernel_width = 5
    input_time_emb_shape = [batches, 8]
    norm_groups = 2
    eps = 0.01
    shard_count = 2
    theta = make_resnet_block_2d_theta(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        input_time_emb_channels=input_time_emb_shape[1],
    )
    theta.rename_tensors_to_paths()
    input_image = torch.rand(
        batches,
        in_channels,
        height,
        width,
    )
    input_time_emb = torch.rand(input_time_emb_shape)

    sharding_spec = ResnetBlock2DSplitOutputChannelsSharding(shard_count=shard_count)
    sharded_theta = ops.reshard(theta, sharding_spec)

    # Roundtrip the dataset, which anchors the tensors as parameters to be loaded
    # vs constants to be frozen (TODO: This is a bit wonky).
    ds = Dataset({}, sharded_theta)
    ds.save(args.parameters)
    ds = Dataset.load(args.parameters)

    sharded_resnet_block = ResnetBlock2D(
        theta=ds.root_theta,
        groups=norm_groups,
        eps=eps,
        non_linearity="relu",
        output_scale_factor=None,
        dropout=0.0,
        temb_channels=input_time_emb_shape[1],
        time_embedding_norm="default",
    )
    sharded_input_image = ops.reshard_split(input_image, dim=1, count=shard_count)
    sharded_input_time_emb = ops.replicate(input_time_emb, count=shard_count)
    expected_result = sharded_resnet_block(sharded_input_image, sharded_input_time_emb)

    exported_resnet_block = aot.export(
        sharded_resnet_block,
        args=(
            sharded_input_image,
            sharded_input_time_emb,
        ),
    )
    exported_resnet_block.save_mlir(args.mlir)
    target_backend = "hip"
    for i in range(shard_count):
        exported_resnet_block.session.set_flags(
            f"--iree-hal-target-device={target_backend}[{i}]",
            "--iree-rocm-target-chip=gfx942",
        )

    print("Compiling module ...")
    exported_resnet_block.compile(save_to=args.module, target_backends=None)

    actual_result = run_iree_module(
        sharded_input_image=sharded_input_image,
        sharded_input_time_emb=sharded_input_time_emb,
        module_path=args.module,
        parameters_path=args.parameters,
    )
    for actual_shard, expected_shard in zip(
        actual_result.shards, expected_result.shards
    ):
        torch.testing.assert_close(
            unbox_tensor(actual_shard), unbox_tensor(expected_shard)
        )


if __name__ == "__main__":
    main(sys.argv[1:])
