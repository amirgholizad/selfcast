"""selfcast: turn text into a podcast in your own cloned voice, locally (Chatterbox TTS)."""
import os

# Must be set before torch / librosa / tqdm / transformers are imported anywhere.
for _key, _value in {
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",  # CPU fallback for any op the Apple GPU lacks
    "TQDM_DISABLE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_VERBOSITY": "error",
    "HF_HUB_VERBOSITY": "error",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
}.items():
    os.environ.setdefault(_key, _value)
