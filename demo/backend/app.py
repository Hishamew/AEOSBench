__all__ = [
    'app',
    'CACHE_ROOT',
]
import fcntl
import os
import pathlib
import sys
from typing import Optional

from flask import Flask

app = Flask(__name__)

# --------------------------
# Path Configuration (Ensure cache directory exists)
# --------------------------
# Define cache root directory and convert to absolute path
CACHE_ROOT: pathlib.Path = pathlib.Path(__file__).parent.parent / '.cache'
# Create cache directory if not exists (support multi-level directories)
CACHE_ROOT.mkdir(exist_ok=True, parents=True)
