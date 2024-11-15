import sys
from pathlib import Path
import os

os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
