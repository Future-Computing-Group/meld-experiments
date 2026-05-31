import os
import sys

# make `src/` importable as top-level packages (smp, ...)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
