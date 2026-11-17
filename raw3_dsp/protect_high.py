"""Test noisy-input routing between the current raw3 and alternate DSP settings."""
import json
from pathlib import Path

import numpy as np
from scipy import signal

from . import raw3_next
from . import transient_voice

HERE = Path(__file__).resolve().parent
BRANCH = raw3_next.branch
ALTERNATES = {
    "speech": dict(scale=1.12, pause_floor=.24, wave=raw3_next.GENTLE["wave"] | dict(scale=35)),
    "tracking": dict(rate=.48, quantile=24, quantile_mix=.30, adaptive_scale=(1.18, 1.52)),
    "speech_stronger": dict(scale=.96, pause_floor=.33, wave=raw3_next.GENTLE["wave"] | dict(scale=29)),
}


def load_settings():
    return tuple(json.loads((HERE / path).read_text()) for path in
                 ("next_recommended.json", "transient_voice_recommended.json"))


def baseline(noisy, sr, settings, noise_refiner=None, noise_reference=None):
    raw3, transient = settings
    if len(noisy) < max(2048, round(.08 * sr)) or not np.any(noisy):
        y = raw3_next.denoise(noisy, sr, raw3)
        return y, y
    context = transient_voice.prepare(noisy, sr, raw3, noise_refiner, noise_reference)
    return context["base"], transient_voice.apply(context, transient)


def features(noisy, pilot, sr):
    sos = signal.butter(3, (160, min(3900, .45 * sr)), fs=sr, btype="bandpass", output="sos")
    x, y = signal.sosfiltfilt(sos, np.stack([noisy, pilot]))
    residual_ratio = 10 * np.log10((np.mean(y**2) + 1e-12) / (np.mean((x-y)**2) + 1e-12))
    frame = round(.024 * sr)
    energy = np.mean(x[:len(x)//frame*frame].reshape(-1, frame)**2, axis=1)
    low, high = np.percentile(energy, [25, 85])
    frame_contrast = 10 * np.log10((high + 1e-12) / (low + 1e-12))
    return {"residual": float(residual_ratio), "mixed": float(.7 * residual_ratio + .3 * frame_contrast)}


def weight(features, config):
    return config["blend"] * float(np.clip((config["threshold"] - features[config["guide"]]) / 3, 0, 1))


def alternate(noisy, sr, method, settings, noise_refiner=None, noise_reference=None):
    previous = BRANCH.EXTRA
    try:
        BRANCH.EXTRA = raw3_next.GENTLE | ALTERNATES[method]
        if noise_refiner is not None:
            BRANCH.EXTRA["noise_refiner"] = noise_refiner
        pilot = BRANCH.denoise_channel(noisy, sr, noise_reference)
    finally:
        BRANCH.EXTRA = previous
    return raw3_next.apply(pilot, noisy, sr, settings[0])


def blend(current, raw3, alternative, amount):
    return np.clip(current + amount * (alternative - raw3), -1, 1 - 1/32768)


def denoise(audio, sr, config, settings=None):
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or sr < 8000:
        raise ValueError("Expected finite mono audio at 8 kHz or above")
    if settings is None:
        settings = load_settings()
    raw3, current = baseline(x, sr, settings)
    if not config or len(x) < 2048 or not np.any(x):
        return current
    amount = weight(features(x, raw3, sr), config)
    if not amount:
        return current
    return blend(current, raw3, alternate(x, sr, config["method"], settings), amount)
