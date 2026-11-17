"""Tune raw3 spectral shaping and residual tone suppression using noisy audio."""
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from . import raw3_siren_estimator as siren

HERE = Path(__file__).resolve().parent
EPS = 1e-12


def tone_config():
    return json.loads((HERE / "raw3_noise_siren_local_config.json").read_text())


def prepare(noisy, sr, settings, tone, need_tone=True, need_spectrum=True):
    current = siren.prepare(noisy, sr, settings)
    base = current["base"]
    if len(noisy) < max(2048, round(.08 * sr)) or not np.any(noisy):
        return dict(base=base, short=True)

    correction = np.zeros_like(base)
    if need_tone:
        analysis = siren.analyze(noisy, sr, base)
        if analysis is not None and siren.enabled(analysis["cues"], tone):
            changed = siren.alternate(noisy, sr, current, settings, "line")
            correction = siren.tone_delta(base, changed, analysis, sr)
    context = dict(base=base, correction=correction)
    return prepare_spectrum(noisy, sr, context) if need_spectrum else context


def prepare_spectrum(noisy, sr, context):
    if context.get("short"):
        return context
    base, correction = context["base"], context["correction"]
    window = 2 * round(.032 * sr)
    spec = dict(fs=sr, nperseg=window, noverlap=window-window//4)
    f, t, before = signal.stft(base, **spec)
    _, _, delta = signal.stft(correction, **spec)
    _, _, raw = signal.stft(noisy, **spec)
    p, q = np.abs(raw)**2, np.abs(before)**2
    band = (f >= 200) & (f <= min(3900, .45 * sr))
    frame = q[band].sum(axis=0)
    full = np.flatnonzero((t * sr >= window / 2) & (t * sr + window / 2 <= len(base)))
    quiet = full[np.argsort(frame[full])[:max(1, round(.20 * len(full)))]]
    noise = np.median(p[:, quiet], axis=1) / np.log(2)
    noise = uniform_filter1d(noise, 5, mode="nearest")
    observed = uniform_filter1d(uniform_filter1d(p, 5, axis=0), 3, axis=1)
    relative = 10 * np.log10((observed + EPS) / (noise[:, None] + EPS))
    speech = uniform_filter1d(np.clip((relative - 2.) / 11., 0, 1), 5, axis=1, mode="nearest")
    loudness = 10 * np.log10((frame + EPS) / (np.median(frame[quiet]) + EPS))
    pause = 1 - np.clip((loudness - 1.) / 10., 0, 1)
    pause = uniform_filter1d(pause, 7, mode="nearest")
    return context | dict(spec=spec, spectrum=before, tone_spectrum=delta, f=f, speech=speech, pause=pause)
