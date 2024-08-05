import random
from typing import List, Optional, Tuple, Union

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.utils import get_max_shared_memory_bytes, is_hip

default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float: 1e-5}
default_rtol = {
    torch.float16: 1e-3,
    torch.bfloat16: 1.6e-2,
    torch.float: 1.3e-6
}


def get_default_atol(output) -> float:
    return default_atol[output.dtype]


def get_default_rtol(output) -> float:
    return default_rtol[output.dtype]

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_BLOCKS = 4321  # Arbitrary values for testing
PARTITION_SIZE = 512
# flshattF and tritonflashattF supported: {torch.float16, torch.bfloat16}
DTYPES = [torch.half, torch.bfloat16]
NUM_GEN_SEQS = [7]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
HEAD_SIZES = [64, 80, 96, 112, 128]

BLOCK_SIZES = [16] #, 32]
USE_ALIBI = [False] #, True]
KV_CACHE_DTYPE = ["auto","fp8"]
SEEDS = [0]
CUDA_DEVICES = ["cuda:0"]
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
def get_kv_cache_torch_dtype(
        cache_dtype: Optional[Union[str, torch.dtype]],
        model_dtype: Optional[Union[str, torch.dtype]] = None) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype

def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=torch_dtype,
                                device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        else:
            raise ValueError(
                f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=torch_dtype,
                                  device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        else:
            raise ValueError(
                f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches

def test_paged_attention(
    num_heads: Tuple[int, int],
    head_size: int,
) -> None:
    seed = 2024
    device = "cuda"
    dtype = torch.half
    kv_cache_dtype = "auto"
    block_size = 16
    use_alibi = False
    num_seqs = 4
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float)

    seq_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables_lst: List[List[int]] = []
    for _ in range(num_seqs):
        block_table = [
            i for i in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)

    block_tables = torch.tensor(block_tables_lst, dtype=torch.int)

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches_with_random(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    kv_scale = 1.0

    # Call the paged attention kernel.
    output1 = torch.empty_like(query)
    num_partitions = ((max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
    assert PARTITION_SIZE % block_size == 0
    num_seqs, num_heads, head_size = output1.shape
    tmp_output = torch.empty(
        size=(num_seqs, num_heads, num_partitions, head_size),
        dtype=output1.dtype,
    )
    exp_sums = torch.empty(
        size=(num_seqs, num_heads, num_partitions),
        dtype=torch.float32,
    )
    max_logits = torch.empty_like(exp_sums)
    ops.paged_attention_v2(
        output1,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        kv_scale,
    )
    

    # Run the reference implementation.
    if kv_cache_dtype == "fp8":
        # Convert cache data back to dtype.
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        key_cache_shape = (NUM_BLOCKS, num_kv_heads, head_size // x,
                           block_size, x)
        dequantized_key_cache = torch.empty(size=key_cache_shape,
                                            dtype=dtype,
                                            device=device)
        ops.convert_fp8(dequantized_key_cache, key_cache)
        key_cache = dequantized_key_cache

        value_cache_shape = value_cache.shape
        dequantized_value_cache = torch.empty(size=value_cache_shape,
                                              dtype=dtype,
                                              device=device)
        ops.convert_fp8(dequantized_value_cache, value_cache)
        value_cache = dequantized_value_cache

    origin_block_table_lst = block_tables_lst
    block_tables_lst: List[List[int]] = []
    for _ in range(num_seqs):
        block_table = [(255<<24) + 256 * i for i in range(max_num_blocks_per_seq//256)]
        if (max_num_blocks_per_seq % 256) != 0:
            re = max_num_blocks_per_seq % 256
            block_table.append(((re-1)<<24)+max_num_blocks_per_seq-re)
        block_tables_lst.append(block_table)
    
    # check if the block tables are consistant
    for i in range(max_num_blocks_per_seq):
        origin_phy_block_id = origin_block_table_lst[0][i]
        start_logic_id = 0
        for entry in block_tables_lst[0]:
            start_phy_id = (entry & ((1<<24)-1))
            entry_size = (entry >> 24) + 1
            if (start_logic_id + entry_size) > i:
                phy_block_id = (i - start_logic_id) + start_phy_id
                break
            start_logic_id += entry_size
        assert origin_phy_block_id == phy_block_id

    block_tables = torch.tensor(block_tables_lst, dtype=torch.uint32)
    output2 = torch.empty_like(query)
    ops.dy_paged_attention(
        output2,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        kv_scale,
    )

    # NOTE(woosuk): Due to the kernel-level differences in the two
    # implementations, there is a small numerical difference in the two
    # outputs. Thus, we use a relaxed tolerance for the test.
    atol = get_default_atol(output1) if is_hip() else 1e-3
    rtol = get_default_rtol(output1) if is_hip() else 1e-5

    # NOTE(zhaoyang): FP8 KV Cache will introduce quantization error,
    # so we use a relaxed tolerance for the test.
    atol, rtol = 1e-3, 1e-5
    if kv_cache_dtype == "fp8":
        atol, rtol = 1e-2, 1e-5
    print(torch.allclose(output1, output2, atol=atol, rtol=rtol))

for num_heads in NUM_HEADS:
    for head_size in HEAD_SIZES:
        print(f"testing num_heads = {num_heads} head_size = {head_size}")
        test_paged_attention(num_heads, head_size)