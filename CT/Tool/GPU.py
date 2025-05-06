# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

import os
import numpy

os.system('nvidia-smi -q -d Memory | grep -A8 "GPU" | grep "Free" | awk \'{print $3}\' > tmp')

with open('tmp', 'r') as f:
    lines = f.readlines()

if not lines:
    raise ValueError("tmp file is empty, no GPU information found")

gpu_free_memory = [int(x.strip()) for x in lines]
gpu_free_memory_without_gpu4 = gpu_free_memory[:4] + gpu_free_memory[5:]

max_gpu_index = numpy.argmax(gpu_free_memory_without_gpu4)

print(f"Selected GPU index: {max_gpu_index}")

os.environ['CUDA_VISIBLE_DEVICES'] = str(max_gpu_index)

os.system('rm tmp')