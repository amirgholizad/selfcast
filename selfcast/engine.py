"""Thin wrapper over the three Chatterbox engines: offline-first loading, voice
conditioning that is computed once and cached per voice, and quiet synthesis."""
from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from .voices import Voice

for _name in ("transformers", "huggingface_hub", "chatterbox", "diffusers"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# repo, files to fetch, files that must be present for an offline load
MODELS = {
    "turbo": ("ResembleAI/chatterbox-turbo",
              ["ve.safetensors", "t3_turbo_v1.safetensors", "s3gen_meanflow.safetensors",
               "*.json", "merges.txt", "conds.pt"],
              ["ve.safetensors", "t3_turbo_v1.safetensors", "s3gen_meanflow.safetensors",
               "vocab.json", "merges.txt"]),
    "standard": ("ResembleAI/chatterbox",
                 ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"],
                 ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]),
    "multilingual": ("ResembleAI/chatterbox",
                     ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                      "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
                     ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                      "grapheme_mtl_merged_expanded_v1.json"]),
}


@contextlib.contextmanager
def _quiet():
    """Swallow the libraries' chatty prints and warnings."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def checkpoint_dir(repo: str, patterns: list[str], required: list[str], log, what: str) -> Path:
    """Local snapshot of a Hugging Face repo: the offline cache when complete, else download."""
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(repo_id=repo, allow_patterns=patterns, local_files_only=True))
        if all((path / f).exists() for f in required):
            return path
    except Exception:
        pass
    log(f"Downloading {what} (first run only)...")
    return Path(snapshot_download(repo_id=repo, allow_patterns=patterns))


def _checkpoint_dir(engine: str, log) -> Path:
    return checkpoint_dir(*MODELS[engine], log=log, what=f"the {engine} model weights (a few GB)")


def _model_class(engine: str):
    with _quiet():
        if engine == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS as cls
        elif engine == "standard":
            from chatterbox.tts import ChatterboxTTS as cls
        else:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS as cls
    return cls


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_power_mode(mode: str, device: str) -> None:
    from . import config as C

    if mode == "eco":
        if hasattr(os, "nice"):  # not on Windows
            with contextlib.suppress(OSError):
                os.nice(C.ECO_NICE)  # lower priority: everything else on the machine goes first
        torch.set_num_threads(C.ECO_THREADS)
        if device == "mps":
            torch.mps.set_per_process_memory_fraction(C.ECO_MPS_MEMORY_FRACTION)


def peak_memory_gb() -> float:
    """Lifetime peak memory of this process, Apple GPU buffers included: the "peak memory
    footprint" that `/usr/bin/time -l` reports. Falls back to peak RSS on other systems."""
    if sys.platform == "darwin":
        import ctypes

        class _RUsageInfoV4(ctypes.Structure):  # <sys/resource.h>: rusage_info_v4
            _fields_ = [("uuid", ctypes.c_uint8 * 16), ("fields", ctypes.c_uint64 * 35)]

        with contextlib.suppress(OSError, AttributeError):
            info = _RUsageInfoV4()
            if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) == 0:
                return info.fields[28] / 1e9  # ri_lifetime_max_phys_footprint
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9


class Engine:
    def __init__(self, name: str, device: str | None = None, power: str = "eco", log=print):
        if name not in MODELS:
            raise ValueError(f"unknown engine '{name}'")
        self.name = name
        self.device = device or pick_device()
        self.peak_gpu_gb = 0.0
        apply_power_mode(power, self.device)
        ckpt = _checkpoint_dir(name, log)
        with _quiet():
            self.model = _model_class(name).from_local(ckpt, self.device)
        self.sr: int = self.model.sr

    def use_voice(self, voice: Voice) -> str:
        """Load the voice's cached conditioning for this engine, computing it on first use.
        Returns a short hash identifying the conditioning (used as a cache key)."""
        path = voice.conds(self.name)
        conds_cls = sys.modules[type(self.model).__module__].Conditionals
        if path.exists():
            self.model.conds = conds_cls.load(path, map_location="cpu").to(self.device)
        else:
            with _quiet(), torch.inference_mode():
                self.model.prepare_conditionals(str(voice.reference))
            self.model.conds.to("cpu").save(path)
            self.model.conds.to(self.device)
        return hashlib.sha1(path.read_bytes()).hexdigest()[:12]

    def synthesize(self, text: str, seed: int, *, language: str = "en", exaggeration: float = 0.5,
                   cfg_weight: float = 0.5, temperature: float = 0.8) -> np.ndarray:
        torch.manual_seed(seed)
        with _quiet(), torch.inference_mode():
            if self.name == "turbo":  # turbo ignores exaggeration / cfg
                wav = self.model.generate(text, temperature=temperature)
            elif self.name == "standard":
                wav = self.model.generate(text, exaggeration=exaggeration, cfg_weight=cfg_weight,
                                          temperature=temperature)
            else:
                wav = self.model.generate(text, language_id=language, exaggeration=exaggeration,
                                          cfg_weight=cfg_weight, temperature=temperature)
        out = wav.squeeze(0).float().cpu().numpy()
        if self.device == "mps":
            self.peak_gpu_gb = max(self.peak_gpu_gb, torch.mps.driver_allocated_memory() / 1e9)
            torch.mps.empty_cache()
        return out

    def speaker_embedding(self, wav: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        ve_sr = self.model.ve.hp.sample_rate
        if sr != ve_sr:  # resample here; the encoder's own path needs the optional 'resampy'
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=ve_sr)
        with _quiet(), torch.inference_mode():
            emb = np.asarray(self.model.ve.embeds_from_wavs([wav.astype(np.float32)], sample_rate=ve_sr))
        emb = emb.reshape(-1, emb.shape[-1]).mean(axis=0)
        return emb / (np.linalg.norm(emb) + 1e-9)
