# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/data', 'disk0', 'Workspace', 'Compiler-Toolchain', 'Compiler-Toolchain')))
from CT.Manager.Path.path import Path

path = Path()
print(path.path_selected())