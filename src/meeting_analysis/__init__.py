"""Vietnamese meeting audio analysis system (Bài toán 2).

Importing this package configures the Hugging Face cache so model weights are
stored inside the project. The default ``~/.cache/huggingface`` may be
unwritable in restricted environments, which otherwise surfaces as a confusing
``PermissionError`` deep inside a model download.
"""

from .config import configure_hf_home

__version__ = "0.1.0"

configure_hf_home()
