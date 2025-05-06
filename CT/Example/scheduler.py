# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

import os
import sys
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain')))
from CT.Scheduler.simpleScheduler import SimpleScheduler

simpleScheduler = SimpleScheduler()