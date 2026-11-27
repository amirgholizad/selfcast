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


# --------------------------------------------------------------------------- fixed-shape decoding

CACHE_BUCKET = 256  # static KV cache lengths are rounded up to this, so most chunks share one size
S3GEN_BUCKET = 50   # S3Gen input is padded with silence to a multiple of this many tokens (2 s)


def _static_cache(t3, length: int):
    """One StaticCache per model, reused (cleared) for every chunk, so every chunk decodes with
    the same tensor shapes. It's only replaced when a chunk needs a longer one.

    StaticCache writes new keys/values with index_copy_, which the Apple GPU lacks: with the CPU
    fallback on, every step copied the whole cache to the CPU and back (~115 ms per token). So
    each layer writes with plain index assignment instead, which runs on the GPU."""
    from transformers.cache_utils import StaticCache

    cache = getattr(t3, "_selfcast_cache", None)
    if cache is not None and cache.max_cache_len >= length:
        cache.reset()
        return cache
    if cache is not None:  # break the layer <-> update cycles so the old tensors are freed now
        for layer in cache.layers:
            layer.__dict__.pop("update", None)
        t3._selfcast_cache = cache = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    cache = StaticCache(config=t3.tfmr.config, max_cache_len=length)
    for layer in cache.layers:
        def update(key_states, value_states, cache_kwargs=None, layer=layer):
            if not layer.is_initialized:
                layer.lazy_initialization(key_states, value_states)
            pos = cache_kwargs["cache_position"]
            layer.keys[:, :, pos] = key_states
            layer.values[:, :, pos] = value_states
            return layer.keys, layer.values
        layer.update = update
    t3._selfcast_cache = cache
    return cache


def _s3gen_bucketed(inference, silence_token: int):
    """Wrap S3Gen.inference so its input length is always a multiple of S3GEN_BUCKET tokens
    (padded with the silence token), and cut the padding off the audio again. An episode then
    uses a handful of input shapes instead of a new one per chunk, so PyTorch's per-shape graph
    cache on Apple GPUs stops growing."""
    def wrapped(speech_tokens, *args, **kwargs):
        n = speech_tokens.size(-1)
        target = -(-n // S3GEN_BUCKET) * S3GEN_BUCKET
        if target > n:
            pad = torch.full((*speech_tokens.shape[:-1], target - n), silence_token,
                             dtype=speech_tokens.dtype, device=speech_tokens.device)
            speech_tokens = torch.cat([speech_tokens, pad], dim=-1)
        wav, source = inference(speech_tokens, *args, **kwargs)
        return wav[..., : round(wav.size(-1) * n / target)], source
    return wrapped


def _turbo_inference_fixed_shapes(self, t3_cond, text_tokens, temperature=0.8, top_k=1000, top_p=0.95,
                                  repetition_penalty=1.2, max_gen_len=1000):
    """Drop-in replacement for chatterbox's T3.inference_turbo with the same sampling, but every
    decoding step runs on tensors of the same shape.

    The original grows its key/value cache and its repetition-penalty input by one token per step.
    On Apple GPUs PyTorch compiles and keeps a graph for every new tensor shape, and never frees
    them: over an episode that added ~5 GB of memory outside the GPU allocator. Here the cache is
    a preallocated StaticCache (attention masks the unused tail), and the repetition penalty uses
    a fixed-size "seen tokens" mask, which gives the same result as penalizing each generated token.
    """
    import torch.nn.functional as F
    from transformers.generation.logits_process import (LogitsProcessorList, TemperatureLogitsWarper,
                                                        TopKLogitsWarper, TopPLogitsWarper)

    warpers = LogitsProcessorList()
    if temperature > 0 and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_k > 0:
        warpers.append(TopKLogitsWarper(top_k))
    if top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p))

    def penalize(logits, seen):
        if repetition_penalty == 1.0:
            return logits
        hit = torch.where(logits < 0, logits * repetition_penalty, logits / repetition_penalty)
        return torch.where(seen, hit, logits)

    def sample(logits, seen):
        # same order as the original: temperature, top-k, top-p, then the repetition penalty
        logits = penalize(warpers(None, logits), seen)
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    start = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _ = self.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=start,
                                          cfg_weight=0.0)
    prompt_len = embeds.size(1)
    cache_len = -(-(prompt_len + max_gen_len + 1) // CACHE_BUCKET) * CACHE_BUCKET
    cache = _static_cache(self, cache_len)
    out = self.tfmr(inputs_embeds=embeds, past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(prompt_len, device=embeds.device))
    logits = self.speech_head(out[0][:, -1:])[:, -1, :]

    vocab = logits.size(-1)
    seen = torch.zeros(1, vocab, dtype=torch.bool, device=logits.device)
    seen[0, start[0, 0]] = start[0, 0] < vocab  # the original penalizes the start token on step one
    token = sample(logits, seen)
    seen = torch.zeros_like(seen)
    tokens = [token]
    position = torch.tensor([prompt_len], device=embeds.device)
    for _ in range(max_gen_len):
        seen.scatter_(1, token, True)
        out = self.tfmr(inputs_embeds=self.speech_emb(token), past_key_values=cache, use_cache=True,
                        cache_position=position)
        position += 1
        logits = self.speech_head(out[0])[:, -1, :]
        if torch.all(torch.isinf(logits) & (logits < 0)):
            break
        token = sample(logits, seen)
        tokens.append(token)
        if torch.all(token == self.hp.stop_speech_token):
            break
    all_tokens = torch.cat(tokens, dim=1)
    if all_tokens.size(1) > 0 and all_tokens[0, -1] == self.hp.stop_speech_token:
        all_tokens = all_tokens[:, :-1]
    return all_tokens


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
        if name == "turbo":
            import types

            from chatterbox.models.s3gen.const import S3GEN_SIL

            self.model.t3.inference_turbo = types.MethodType(_turbo_inference_fixed_shapes, self.model.t3)
            self.model.s3gen.inference = _s3gen_bucketed(self.model.s3gen.inference, S3GEN_SIL)
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
