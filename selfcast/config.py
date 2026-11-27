"""Paths and tunable defaults."""
import os
from pathlib import Path

# Voices, podcasts and caches live in the repo folder by default; set SELFCAST_HOME to move them.
ROOT = Path(os.environ.get("SELFCAST_HOME") or Path(__file__).resolve().parent.parent).expanduser()
VOICES_DIR = ROOT / "voices"
PODCASTS_DIR = ROOT / "podcasts"
SCRIPTS_DIR = ROOT / "scripts"
CACHE_DIR = ROOT / ".cache" / "chunks"
USER_LEXICON = ROOT / "pronunciations.toml"  # your own pronunciations (git-ignored)

ENGINES = ("turbo", "standard", "multilingual")
DEFAULT_ENGINE = "turbo"  # fastest on Apple Silicon; English only
MULTILINGUAL_LANGUAGES = (
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
)

# Reference (voice sample) preparation
REF_SR = 24000
MIN_REF_SECONDS = 6.0     # Turbo refuses anything under 5 s of speech
GOOD_REF_SECONDS = 15.0   # below this we warn that likeness may suffer
MAX_REF_SECONDS = 60.0    # more than this adds nothing
REF_SCAN_SECONDS = 300.0  # --ref-start auto looks for the best window in this much speech
REF_LUFS = -20.0

# Power / memory. "eco" (default) trades speed for a cool, responsive laptop:
# low CPU priority, 2 threads, a GPU-memory cap and a short rest between chunks.
POWER_MODES = ("eco", "fast")
DEFAULT_POWER = "eco"
ECO_THREADS = 2
ECO_NICE = 10
ECO_REST_RATIO = 0.5          # rest 0.5 s for every 1 s spent generating
ECO_MPS_MEMORY_FRACTION = 0.6  # of macOS's recommended GPU working set (~10.7 GB on 16 GB Macs)
HEAVY_ENGINES = ("standard", "multilingual")  # peaked at ~18 GB on a 16 GB M4 (turbo: ~4-5 GB)

# Whisper QA: every chunk is transcribed (openai/whisper-base.en, on the CPU) and compared
# with its text; a take whose word error rate is above the threshold is re-taken.
QA_DEFAULT = True
QA_THRESHOLD = 0.1

# Output
PODCAST_LUFS = -16.0      # standard podcast loudness target
WORDS_PER_MINUTE = 150    # rough speaking rate, for estimates only

PREVIEW_TEXT = (
    "Hi, this is a preview of my cloned voice. I'm reading a short passage so I can "
    "hear how natural it sounds, and decide whether it's ready for my podcast."
)
