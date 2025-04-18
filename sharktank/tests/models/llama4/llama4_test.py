from parameterized import parameterized
import transformers.models
from sharktank.utils.testing import TempDirTestBase
from sharktank.models.llama4.testing import (
    make_toy_model_config,
    config_to_hugging_face_text_config,
    theta_to_hugging_face_state_dict,
)
from sharktank.models.llama.testing import make_random_llama_theta
from sharktank.models.llm import PagedLlmModelV1
import transformers

import torch


def convert_hf_2D_input_mask_to_4D_attention_mask(
    mask: torch.Tensor, model: PagedLlmModelV1
) -> torch.Tensor:
    inverted_mask = mask == 0
    return model.attention_mask(inverted_mask)


class Llama4Test(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)

    @parameterized.expand(
        [
            (torch.float32, 1e-5),
        ]
    )
    def testCompareToyEagerVsHuggingFace(self, dtype: torch.dtype, atol: float):
        torch.set_printoptions(linewidth=88, threshold=1000, edgeitems=3, sci_mode=True)

        config = make_toy_model_config(dtype=dtype)
        theta = make_random_llama_theta(config, dtype=dtype)
        hf_config = config_to_hugging_face_text_config(config)

        model = PagedLlmModelV1(theta=theta, config=config)
        hf_model = transformers.models.llama4.Llama4ForCausalLM(hf_config)

        orig_state_dict = hf_model.state_dict()
        hf_state_dict = theta_to_hugging_face_state_dict(theta, config)
        hf_model.load_state_dict(hf_state_dict)

        batch_size = 41
        batch_seq_len = config.hp.context_length
        input_ids = torch.randint(
            low=0,
            high=config.vocabulary_size,
            size=[batch_size, batch_seq_len],
            dtype=torch.long,
        )
        # inputs_embeds = torch.rand(size=[batch_size, batch_seq_len, config.hp.embedding_length], dtype=dtype)
        # We need to create the cache ourselves as HF would create it always in bf16.
        hf_past_key_values = transformers.cache_utils.HybridChunkedCache(
            hf_config,
            max_batch_size=input_ids.shape[0],
            max_cache_len=input_ids.shape[1],
            dtype=dtype,
        )

        hf_2d_attention_mask = torch.randint_like(input_ids, low=0, high=2)
        attention_mask = convert_hf_2D_input_mask_to_4D_attention_mask(
            mask=hf_2d_attention_mask, model=model
        )

        from sharktank.utils.patching import SaveModuleResultTensorsPatch

        hf_intermediates_saver = SaveModuleResultTensorsPatch(with_before_forward=True)
        hf_intermediates_saver.patch_child_modules(hf_model)

        hf_output = hf_model(
            input_ids=input_ids,
            attention_mask=hf_2d_attention_mask,
            past_key_values=hf_past_key_values,
        )

        page_count = (len(input_ids[0]) // config.block_seq_stride) * batch_size
        kv_cache_state = model.cache.allocate(page_count)
        seq_block_ids = torch.arange(
            start=0, end=input_ids.numel() // config.block_seq_stride, dtype=torch.long
        ).view(batch_size, batch_seq_len // config.block_seq_stride)

        intermediates_saver = SaveModuleResultTensorsPatch(with_before_forward=True)
        intermediates_saver.patch_child_modules(model)

        model.prefill(
            tokens=input_ids,
            attention_mask=attention_mask,
            cache_state=kv_cache_state,
            seq_block_ids=seq_block_ids,
        )

        hf_intermediates_saver.save_file(
            "hf_trace.safetensors", skip_unsupported_dtypes=True
        )
        intermediates_saver.save_file("trace.safetensors", skip_unsupported_dtypes=True)
