"""Apply the verified classical raw3 equalizer to a single channel."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy import signal

from . import raw3_pesq_only as raw3

HERE = Path(__file__).resolve().parent
EPS = 1e-12


@lru_cache(maxsize=1)
def settings():
    config = json.loads((HERE / "traditional_pesq_narrow_config.json").read_text())
    return raw3.siren.settings(), raw3.tone_config(), config


def rms(x):
    return np.sqrt(np.mean(x**2) + EPS)

def limit(y):
    peak = np.max(np.abs(y), initial=0.)
    return y * (.999 / peak) if peak > .999 else y

def gaussian(freqs, center, width):
    return np.exp(-.5*(np.log2(np.maximum(freqs, 1.)/center)/width)**2)

def voice_center(ctx):
    if "voice_center" not in ctx:
        f = ctx["f"]
        region = (f>=265)&(f<=720)
        values = np.abs(ctx["spectrum"][region])**2
        quiet, active = np.percentile(values,[25,85],axis=1)
        speech = np.maximum(active - 1.5*quiet,0.)
        if float(speech.sum()) <= EPS:
            ctx["voice_center"] = 450.
        else:
            ctx["voice_center"] = float(np.exp(np.dot(speech,np.log(f[region]))/speech.sum()))
    return ctx["voice_center"]

def equalized(ctx, config):
    f = ctx["f"]
    cfg = config
    center = cfg["center"]
    if cfg.get("steer"):
        center += cfg["steer"]*np.clip(voice_center(ctx)-450.,-94.,94.)
    body = gaussian(f, center, cfg["width"])
    speech = (1-cfg["gate"]) + cfg["gate"]*ctx["speech"]
    db = cfg["boost"] * body[:,None] * speech
    high = np.clip((f-1700)/1600, 0, 1)
    low = np.clip((330-f)/230, 0, 1)
    db += cfg.get("treble",0.) * high[:,None] + cfg.get("bass",0.) * low[:,None]
    if cfg.get("presence"):
        db += cfg["presence"] * gaussian(f,1450.,.58)[:,None]
    output = ctx["spectrum"] + cfg.get("tone",1.) * ctx["tone_spectrum"]
    _, out = signal.istft(output*10**(db/20),**ctx["spec"])
    return out[:len(ctx["base"])]

def denoise_channel(noisy, sample_rate):
    x = np.asarray(noisy, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or sample_rate < 8000:
        raise ValueError("Expected finite mono audio at 8 kHz or above")
    core, tone, config = settings()
    ctx = raw3.prepare(x, sample_rate, core, tone, True, True)
    if ctx.get("short"):
        return ctx["base"]
    y = equalized(ctx, config)
    if config.get("match_rms"):
        y *= rms(ctx["base"]) / rms(y)
    return limit(y * 10**(config.get("level_db", 0.)/20))
