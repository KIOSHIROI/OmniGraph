import os
from typing import Optional

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_CACHE_DIR = "/media/disk/02drive/13hias/.cache"


def setup_env(hf_endpoint: Optional[str] = None, hf_cache_dir: Optional[str] = None) -> None:
    """Configure Hugging Face cache/mirror paths.

    Priority:
    1) Explicit function args
    2) OMNIGRAPH_HF_ENDPOINT / OMNIGRAPH_HF_CACHE env vars
    3) Existing HF_* env vars (if already set)
    4) Project defaults (backward compatible)
    """
    endpoint = (
        hf_endpoint
        or os.environ.get("OMNIGRAPH_HF_ENDPOINT")
        or os.environ.get("HF_ENDPOINT")
        or DEFAULT_HF_ENDPOINT
    )
    cache_dir = (
        hf_cache_dir
        or os.environ.get("OMNIGRAPH_HF_CACHE")
        or os.environ.get("HF_HOME")
        or DEFAULT_HF_CACHE_DIR
    )

    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")
