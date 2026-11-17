"""Estimate persistent and sweeping tonal noise inside raw3's main STFT."""
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter, median_filter, percentile_filter, uniform_filter1d

from . import protect_high
from .tracked_tones import link_peaks, pitch_support

HERE = Path(__file__).resolve().parent
EPS = 1e-12


def settings():
    return (protect_high.load_settings(),
            json.loads((HERE / "protect_high_speech_config.json").read_text()))


def estimate(f, t, power, complete):
    """Learn tracks from quieter frames and carry them through speech."""
    hz = f[1] - f[0]
    if len(complete) < 12:
        return None
    band = (f >= 350) & (f < 3000)
    background = median_filter(power, size=(2 * max(4, round(125 / hz)) + 1, 1), mode="nearest")
    prominence = 10 * np.log10((power + EPS) / (background + EPS))
    spread = background[band].sum(axis=0)
    quiet = spread <= np.percentile(spread[complete], 40)
    tracks = link_peaks(power, prominence, hz, band)
    target = np.zeros_like(power)
    near = np.zeros_like(power)
    support = np.zeros(len(t))
    dt = t[1] - t[0]
    for track in tracks:
        time, bins = np.asarray(track).T
        duration = (time[-1] - time[0] + 1) * dt
        if duration < .19 or len(time) * dt < .75 * duration:
            continue
        anchors = quiet[time] & np.isin(time, complete)
        if anchors.sum() < max(3, round(.15 * len(time))):
            continue
        ridge = gaussian_filter(bins.astype(float), sigma=.75) * hz
        motion = np.ptp(np.percentile(ridge, [10, 90]))
        contrast = float(np.median(prominence[bins[anchors], time[anchors]]))
        if contrast < 10 or (motion < 55 and duration < .40):
            continue
        # Reject choppy peak changes that frequently arise from speech harmonics.
        slope = np.diff(ridge) / np.diff(time)
        if motion >= 55 and np.percentile(np.abs(np.diff(slope)), 72) > 30:
            continue
        idx = np.arange(time[0], time[-1] + 1)
        center = np.interp(idx, time, ridge)
        line = np.maximum(power[bins, time] - background[bins, time], EPS)
        measured = np.interp(idx, time, np.log(line))
        quiet_line = np.interp(idx, time[anchors], np.log(line[anchors]))
        # The stronger of two candidates may be speech. Bound the increase
        # relative to the interpolated noise-only observation.
        amplitude = np.exp(np.minimum(measured, quiet_line + np.log(1.8)))
        width = 1.65 * hz + .55 * np.abs(np.gradient(center))
        width = np.clip(width, 1.65 * hz, 65)
        bump = np.exp(-.5 * ((f[:, None] - center) / width)**2)
        ridge_score = np.clip((contrast - 8) / 11, .22, 1.)
        target[:, idx] = np.maximum(target[:, idx], bump * amplitude[None, :])
        near[:, idx] = np.maximum(near[:, idx], ridge_score * bump)
        support[idx] = np.maximum(support[idx], ridge_score)
    if not np.any(support):
        return None
    near *= band[:, None]
    target *= band[:, None]
    return target, near, support


def refiner(style):
    def build(f, t, power, complete, _reference):
        result = estimate(f, t, power, complete)
        if result is None:
            return None
        target, near, support = result
        if style == "mid":
            residual = np.maximum(power - target, 0)
            floor = percentile_filter(residual, 25, size=(1, 17), mode="nearest")
            floor = 1.55 * uniform_filter1d(floor, 3, axis=0, mode="nearest")
            band = ((f >= 350) & (f < 3000))[:, None]
            target = target + floor
            weight = np.maximum(.82 * near, .56 * band * support[None, :])
        else:
            weight = near

        def replace(previous, frame):
            proposal = target[:, frame]
            if style == "line":
                proposal = np.maximum(proposal, previous)
            return previous + weight[:, frame] * (proposal - previous)

        return replace

    return build


def prepare(noisy, sr, base_settings):
    raw_settings, route_profile = base_settings
    raw3, transient = protect_high.baseline(noisy, sr, raw_settings)
    base = transient
    prior = 0.
    if len(noisy) >= 2048 and np.any(noisy):
        guidance = protect_high.features(noisy, raw3, sr)
        prior = protect_high.weight(guidance, route_profile)
        if prior:
            mild = protect_high.alternate(noisy, sr, "speech", raw_settings)
            base = protect_high.blend(transient, raw3, mild, prior)
    return dict(base=base, prior=prior)


def analyze(noisy, sr, base):
    window = round(.096 * sr)
    if len(noisy) < 4 * window or not np.any(noisy):
        return None
    spec = dict(fs=sr, nperseg=window, noverlap=window-window//4)
    f, t, X = signal.stft(noisy, **spec)
    end = window / (2 * sr)
    complete = np.flatnonzero((t >= end) & (t <= len(noisy) / sr - end))
    result = estimate(f, t, np.abs(X)**2, complete)
    if result is None:
        return None
    _, near, support = result
    _, _, Y = signal.stft(base, **spec)
    energy = np.abs(Y)**2
    band = (f >= 350) & (f < 3000)
    present = energy[band].sum() + EPS
    return dict(cues=dict(audible=float((energy * near).sum() / present),
                          coverage=float(np.mean(support > 0))),
                spec=spec, near=near, f=f, t=t, spectrum=Y)


def audibility(noisy, sr, base):
    analysis = analyze(noisy, sr, base)
    return None if analysis is None else analysis["cues"]


def alternate(noisy, sr, current, base_settings, style):
    raw_settings, _ = base_settings
    adjust = refiner(style)
    raw3, transient = protect_high.baseline(noisy, sr, raw_settings, adjust)
    prior = current["prior"]
    if prior:
        mild = protect_high.alternate(noisy, sr, "speech", raw_settings, adjust)
        transient = protect_high.blend(transient, raw3, mild, prior)
    return transient


def enabled(cues, config):
    return (cues is not None and cues["audible"] >= config["audible"]
            and cues["coverage"] >= config["coverage"])


def mix(current, alternate, amount):
    return (1-amount) * current + amount * alternate


def tone_delta(base, changed, analysis, sr):
    """Keep the main estimator's attenuation only around tracked noise peaks."""
    spec = analysis["spec"]
    f, t, before = analysis["f"], analysis["t"], analysis["spectrum"]
    _, _, after = signal.stft(changed, **spec)
    energy = np.abs(before)**2
    hz = f[1] - f[0]
    local = median_filter(energy, size=(2 * max(4, round(125 / hz)) + 1, 1), mode="nearest")
    prominence = 10 * np.log10((energy + EPS) / (local + EPS))
    pitch, confidence = pitch_support(base, sr, t, energy, hz, prominence)
    multiples = f[:, None] / pitch[None, :]
    distance = np.abs(multiples - np.rint(multiples)) * pitch[None, :]
    voice = confidence[None, :] * np.clip((42 - distance) / 28, 0, 1)
    aligned = np.real(after * np.conjugate(before)) / (energy + EPS)
    attenuation = np.clip(1 - aligned, 0, .78)
    mask = gaussian_filter(analysis["near"] * (1 - .80 * voice), sigma=(.55, .85))
    _, delta = signal.istft(-before * mask * attenuation, **spec)
    delta = delta[:len(base)]
    freq = np.fft.rfftfreq(len(base), 1 / sr)
    taper = .5 - .5 * np.cos(np.pi * np.clip((freq - 350) / 120., 0, 1))
    return np.fft.irfft(np.fft.rfft(delta) * taper, n=len(base))


def denoise(audio, sr, config, base_settings=None):
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or sr < 8000:
        raise ValueError("Expected finite mono audio at 8 kHz or above")
    if base_settings is None:
        base_settings = settings()
    current = prepare(x, sr, base_settings)
    base = current["base"]
    if not config:
        return base
    analysis = analyze(x, sr, base)
    if not enabled(None if analysis is None else analysis["cues"], config):
        return base
    style = config["style"]
    changed = alternate(x, sr, current, base_settings, "line" if style == "local" else style)
    if style == "local":
        changed = base + tone_delta(base, changed, analysis, sr)
    return mix(base, changed, config["mix"])
