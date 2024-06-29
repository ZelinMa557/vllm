import time
import random
from typing import List, Optional

import torch
block_tables_lst: List[List[int]] = []
for _ in range(8):
    block_table = [
        (16 << 24) + random.randint(0, 10)
        for _ in range(4096//(16*32))
    ]
    block_tables_lst.append(block_table)
for block in block_tables_lst:
    for ent in block:
        print(ent)
start_time = time.perf_counter()
for i in range(1000):
    block_tables = torch.tensor(block_tables_lst,
                                dtype=torch.int,
                                device="cuda")
end_time = time.perf_counter()
latency = (end_time - start_time)/1000
print(f"{latency * 1000000:.3f}us")