"""Choose a Torch-capable child interpreter without importing Torch into Qt."""
import sys
from pathlib import Path

def torch_worker_python():
    candidates=[(sys.executable,sys.prefix),(getattr(sys,'_base_executable',sys.executable),sys.base_prefix)]
    for executable,prefix in candidates:
        if (Path(prefix)/'Lib/site-packages/torch/__init__.py').is_file():
            return str(executable)
    # Preserve a clear child import error when neither runtime provides Torch.
    return sys.executable
