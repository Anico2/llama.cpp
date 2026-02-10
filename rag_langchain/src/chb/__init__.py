import os
import sys
from pathlib import Path

os.environ["PROJECT_ROOT"] = str(Path(__file__).parent.parent.parent)
os.environ["CHB_ROOT"] = str(Path(__file__).parent.parent)

sys.path.append(os.environ["CHB_ROOT"])
