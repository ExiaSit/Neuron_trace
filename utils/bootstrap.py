"""Package root and bundled library discovery; no original source-tree imports."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in [ROOT, ROOT/'vendor/pylib', ROOT/'vendor/gcut']:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
os.environ.setdefault('MPLBACKEND', 'Agg')
