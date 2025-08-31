# frust/config.py
from pathlib import Path
import os
try:
    from dotenv import load_dotenv
    load_dotenv(Path.home() / ".env")
except Exception:
    pass

UMA_TOOLS = os.environ.get("UMA_TOOLS")
if not UMA_TOOLS:
    raise RuntimeError("Set UMA_TOOLS in ~/.env to the orca-external-tools path.")