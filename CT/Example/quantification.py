# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

import os
import sys
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain')))
from CT.Quantification.simpleQuantification import SimpleQuantification
from CT.Scheduler.simpleScheduler import SimpleScheduler

model_ids = [
    "Qwen2-7B-Instruct",
    "Qwen2-VL-7B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-VL-7B-Instruct"
]

quantize_batch_size: int = 4

def simpleQuantification(model_id: str, quantize_batch_size: int):
    simpleScheduler = SimpleScheduler
    try: 
        simpleQuantification = SimpleQuantification(
            model_id=model_id,
            quantize_batch_size=quantize_batch_size
        )
    except Exception as e:
        print(f"{model_id} Quantification Error, Reason: {e}")
threads = []

for model_id in model_ids:
    thread = threading.Thread(target=simpleQuantification, args=(model_id, quantize_batch_size))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()