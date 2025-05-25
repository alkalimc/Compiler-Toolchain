# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/data', 'disk0', 'Workspace', 'Compiler-Toolchain', 'Compiler-Toolchain')))
from CT.Manager.Key.key import Key

key = Key()
print(key.keySelected())