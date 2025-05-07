# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

import os
import sys
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/data', 'disk0', 'Workspace', 'Compiler-Toolchain', 'Compiler-Toolchain')))
from CT.Scheduler.simpleScheduler import SimpleScheduler

simpleScheduler = SimpleScheduler()
print(simpleScheduler.gpu_selected)
os.environ["CUDA_VISIBLE_DEVICES"] = str(simpleScheduler.gpu_selected)
print(f"\nCUDA_VISIBLE_DEVICE = {subprocess.run("echo $CUDA_VISIBLE_DEVICES", shell=True, capture_output=True, text=True).stdout}")