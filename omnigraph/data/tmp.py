import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from omnigraph.utils.env import setup_env
setup_env()
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError

def _get_hf_token() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )


token = _get_hf_token()
try:
    # Login using e.g. `huggingface-cli login` or set HF_TOKEN to access this dataset
    ds = load_dataset("laion/relaion400m", token=token)
    print(ds)
except DatasetNotFoundError as exc:
    if "gated dataset" in str(exc).lower() and not token:
        raise RuntimeError(
            "Dataset is gated. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) or run "
            "`huggingface-cli login` and retry."
        ) from exc
    raise