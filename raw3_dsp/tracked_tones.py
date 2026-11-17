"""Input-only narrowband track suppression on the improved Raw3 waveform."""
import math

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from .pitch import estimate_yin

EPS = 1e-12


def stft_options(sr):
    n = 2*round(.022*sr)
    hop = max(1, round(.008*sr))
    nfft = 2**math.ceil(math.log2(2*n))
    return dict(fs=sr, nperseg=n, noverlap=n-hop, nfft=nfft)


def link_peaks(power, prominence, Hz, frequency_range):
    """Connect salient peaks using short-gap prediction and one peak per track."""
    tracks = []
    for frame, column in enumerate(power.T):
        where = signal.find_peaks(prominence[:, frame], height=9, distance=max(2, round(38/Hz)))[0]
        where = where[frequency_range[where] & (column[where] >= column.max()*1e-3)]
        where = where[np.argsort(column[where])[-12:]]
        active = [(i, t) for i, t in enumerate(tracks) if frame-t[-1][0] <= 2]
        links = []
        for i, track in active:
            last_time, last_bin = track[-1]
            step = frame-last_time
            slope = np.clip(last_bin-track[-2][1], -7, 7) if len(track) > 1 else 0
            predicted = last_bin+slope*step
            for peak in where:
                error = abs(peak-predicted)*Hz
                if error < 48+16*(step-1) and abs(peak-last_bin)*Hz < 90*step:
                    links.append((error + 4*(step-1), i, peak))
        seen_track, seen_peak = set(), set()
        for _, i, peak in sorted(links):
            if i not in seen_track and peak not in seen_peak:
                tracks[i].append((frame, int(peak)))
                seen_track.add(i)
                seen_peak.add(peak)
        tracks.extend([[(frame, int(b))] for b in where if b not in seen_peak])
    return tracks


def pitch_support(pilot, sr, time, power, Hz, prominence):
    voice = signal.sosfiltfilt(signal.butter(3, [80, 1350], "bandpass", fs=sr, output="sos"), pilot)
    radius = round(.021*sr)
    support = np.zeros_like(time)
    f0 = np.full_like(time, 100.)
    for j, center in enumerate(np.rint(time*sr).astype(int)):
        segment = voice[max(0, center-radius):min(len(voice), center+radius)]
        hz, confidence = estimate_yin(segment, sr)
        if hz < 65 or confidence <= .64:
            continue
        peaks = signal.find_peaks(prominence[:, j], height=7)[0]
        multiples = peaks*Hz/hz
        useful = (multiples >= 1) & (peaks*Hz < 3000) & (np.abs(multiples-np.rint(multiples)) < .14)
        useful &= power[peaks, j] > np.max(power[:, j])*1e-3
        support[j] = np.clip((confidence-.64)/.25, 0, 1)*np.clip((np.sum(useful)-1)/3, 0, 1)
        f0[j] = hz
    return f0, gaussian_filter1d(support, .85)
