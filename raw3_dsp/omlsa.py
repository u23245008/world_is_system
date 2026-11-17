"""Traditional speech enhancement: IMCRA noise tracking + OM-LSA gain.

Based on I. Cohen, "Noise spectrum estimation in adverse environments: Improved
Minima Controlled Recursive Averaging" (2003) and "Optimal speech enhancement
under signal presence uncertainty using log-spectral amplitude estimator" (2002).

Pipeline (per channel, each file independent):
  sqrt-Hann STFT (75 % overlap)
  -> IMCRA: two-stage minimum tracking -> speech absence prob q -> speech presence prob p
  -> recursive noise PSD update controlled by p
  -> decision-directed a priori SNR (xi)
  -> LSA gain G_H1(xi, gamma), weighted by p:  G = G_H1^p * G_min^(1-p)
  -> weighted overlap-add (sqrt-Hann synthesis)
"""


import numpy as np
from scipy import signal
from scipy.ndimage import convolve1d
from scipy.special import exp1

# ---------------- parameters (time constants are defined for an 8 ms hop
# ---------------- and are rescaled automatically for other hops) ----------------
FRAME_SEC = 0.032        # analysis window length
REF_HOP_SEC = 0.008
ALPHA_S = 0.80           # PSD smoothing over time
ALPHA_D = 0.85           # noise update rate when speech is absent
ALPHA_DD = 0.92          # decision-directed smoothing
XI_MIN_DB = -25.0        # floor of a priori SNR
G_MIN_DB = -18.0         # max attenuation (lower = stronger suppression)
B_MIN = 1.66             # bias of the minimum tracker
GAMMA0 = 4.6             # IMCRA stage-1 thresholds
ZETA0 = 1.67
GAMMA1 = 3.0             # IMCRA stage-2 threshold
Q_MAX = 0.95             # upper bound of speech absence probability
NOISE_BIAS = 1.30        # noise over-estimation factor (beta in the paper: 1.47)
MIN_WIN_SEC = 1.0        # minimum search window length
N_SUB = 8                # number of sub-windows of the minimum tracker
INIT_FRACTION = 0.10     # quietest frames used to initialise the noise PSD

FREQ_KERNEL = np.array([0.25, 0.5, 0.25])


class MinTracker:
    """Running minimum over ~MIN_WIN_SEC using N_SUB sub-windows."""

    def __init__(self, init, sub_len, n_sub):
        self.sub_len = sub_len
        self.buffer = np.full((max(n_sub - 1, 1), len(init)), np.inf)
        self.current = init.copy()
        self.count = 0

    def update(self, value):
        self.current = np.minimum(self.current, value)
        self.count += 1
        result = np.minimum(self.buffer.min(axis=0), self.current)
        if self.count >= self.sub_len:
            self.buffer = np.roll(self.buffer, -1, axis=0)
            self.buffer[-1] = self.current
            self.current = np.full_like(value, np.inf)
            self.count = 0
        return result


def rescale(alpha, hop_sec):
    return alpha ** (hop_sec / REF_HOP_SEC)


def enhance_channel(noisy, sample_rate):
    x = np.asarray(noisy, dtype=np.float64)
    if len(x) < 64 or not np.any(x):
        return x.copy()

    n_fft = int(2 ** np.round(np.log2(FRAME_SEC * sample_rate)))
    n_fft = max(64, min(n_fft, 1 << int(np.floor(np.log2(len(x))))))
    hop = n_fft // 4
    hop_sec = hop / sample_rate
    window = np.sqrt(signal.get_window("hann", n_fft))

    _, times, Y = signal.stft(
        x, fs=sample_rate, window=window, nperseg=n_fft,
        noverlap=n_fft - hop, boundary="zeros", padded=True)
    P = np.abs(Y) ** 2
    n_bins, n_frames = P.shape
    floor = max(P.mean() * 1e-10, 1e-30)

    # ---- initial noise PSD from the quietest complete frames ----
    centers = np.rint(times * sample_rate).astype(int)
    complete = np.flatnonzero(
        (centers >= n_fft // 2) & (centers + n_fft // 2 <= len(x)))
    if not len(complete):
        complete = np.arange(n_frames)
    energy = P[:, complete].mean(axis=0)
    count = max(1, int(np.ceil(INIT_FRACTION * len(complete))))
    chosen = complete[np.argsort(energy)[:count]]
    init_noise = convolve1d(P[:, chosen].mean(axis=1), FREQ_KERNEL, mode="nearest")
    init_noise = np.maximum(init_noise, floor)

    # ---- constants ----
    a_s = rescale(ALPHA_S, hop_sec)
    a_d = rescale(ALPHA_D, hop_sec)
    a_dd = rescale(ALPHA_DD, hop_sec)
    sub_len = max(1, int(round(MIN_WIN_SEC / hop_sec / N_SUB)))
    xi_min = 10 ** (XI_MIN_DB / 10)
    g_min = 10 ** (G_MIN_DB / 20)

    # ---- state ----
    lam = init_noise.copy()                 # noise PSD estimate
    S = init_noise.copy()                   # smoothed PSD, stage 1
    S_t = init_noise.copy()                 # smoothed PSD, stage 2 (speech excluded)
    tracker1 = MinTracker(init_noise / B_MIN, sub_len, N_SUB)
    tracker2 = MinTracker(init_noise / B_MIN, sub_len, N_SUB)
    gain = np.empty_like(P)
    prev_g2_gamma = None

    for l in range(n_frames):
        y2 = P[:, l]

        # ---------- IMCRA stage 1: coarse speech/noise indicator ----------
        S_f = convolve1d(y2, FREQ_KERNEL, mode="nearest")
        S = a_s * S + (1 - a_s) * S_f
        S_min = np.maximum(tracker1.update(S), floor)
        gamma_min = y2 / (B_MIN * S_min)
        zeta = S / (B_MIN * S_min)
        absent = ((gamma_min < GAMMA0) & (zeta < ZETA0)).astype(np.float64)

        # ---------- IMCRA stage 2: smooth only noise-like bins ----------
        num = convolve1d(absent * S_f, FREQ_KERNEL, mode="constant")
        den = convolve1d(absent, FREQ_KERNEL, mode="constant")
        S_new = np.where(den > 1e-9, num / np.maximum(den, 1e-9), S_t)
        S_t = a_s * S_t + (1 - a_s) * S_new
        S_t_min = np.maximum(tracker2.update(S_t), floor)

        gamma_min2 = y2 / (B_MIN * S_t_min)
        zeta2 = S / (B_MIN * S_t_min)
        q = np.where(
            zeta2 < ZETA0,
            np.clip((GAMMA1 - gamma_min2) / (GAMMA1 - 1.0), 0.0, 1.0),
            0.0)
        q = np.minimum(q, Q_MAX)

        # ---------- a priori SNR (decision-directed) ----------
        noise = np.maximum(NOISE_BIAS * lam, floor)
        gamma = y2 / noise
        ml = np.maximum(gamma - 1.0, 0.0)
        if prev_g2_gamma is None:
            xi = np.maximum(ml, xi_min)
        else:
            xi = np.maximum(a_dd * prev_g2_gamma + (1 - a_dd) * ml, xi_min)

        # ---------- LSA gain and speech presence probability ----------
        v = np.maximum(xi / (1 + xi) * gamma, 1e-8)
        g_h1 = np.minimum(xi / (1 + xi) * np.exp(0.5 * exp1(v)), 1.0)
        p = 1.0 / (1.0 + (q / (1.0 - q)) * (1.0 + xi) * np.exp(-v))
        gain[:, l] = g_h1 ** p * g_min ** (1.0 - p)
        prev_g2_gamma = g_h1 ** 2 * gamma

        # ---------- noise PSD update controlled by p ----------
        a_t = a_d + (1 - a_d) * p
        lam = np.maximum(a_t * lam + (1 - a_t) * y2, floor)

    _, enhanced = signal.istft(
        Y * gain, fs=sample_rate, window=window, nperseg=n_fft,
        noverlap=n_fft - hop, boundary=True)
    enhanced = enhanced[:len(x)]
    if len(enhanced) < len(x):
        enhanced = np.pad(enhanced, (0, len(x) - len(enhanced)))
    return np.clip(enhanced, -1, 1)
