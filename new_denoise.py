"""Traditional speech enhancement: IMCRA noise tracking + OM-LSA gain.
2

3
Based on I. Cohen, "Noise spectrum estimation in adverse environments: Improved
4
Minima Controlled Recursive Averaging" (2003) and "Optimal speech enhancement
5
under signal presence uncertainty using log-spectral amplitude estimator" (2002).
6

7
Pipeline (per channel, each file independent):
8
  sqrt-Hann STFT (75 % overlap)
9
  -> IMCRA: two-stage minimum tracking -> speech absence prob q -> speech presence prob p
10
  -> recursive noise PSD update controlled by p
11
  -> decision-directed a priori SNR (xi)
12
  -> LSA gain G_H1(xi, gamma), weighted by p:  G = G_H1^p * G_min^(1-p)
13
  -> weighted overlap-add (sqrt-Hann synthesis)
14
"""
15

16
import argparse
17
from pathlib import Path
18

19
import numpy as np
20
import soundfile as sf
21
from scipy import signal
22
from scipy.ndimage import convolve1d
23
from scipy.special import exp1
24

25
# ---------------- parameters (time constants are defined for an 8 ms hop
26
# ---------------- and are rescaled automatically for other hops) ----------------
27
FRAME_SEC = 0.032        # analysis window length
28
REF_HOP_SEC = 0.008
29
ALPHA_S = 0.80           # PSD smoothing over time
30
ALPHA_D = 0.85           # noise update rate when speech is absent
31
ALPHA_DD = 0.92          # decision-directed smoothing
32
XI_MIN_DB = -25.0        # floor of a priori SNR
33
G_MIN_DB = -18.0         # max attenuation (lower = stronger suppression)
34
B_MIN = 1.66             # bias of the minimum tracker
35
GAMMA0 = 4.6             # IMCRA stage-1 thresholds
36
ZETA0 = 1.67
37
GAMMA1 = 3.0             # IMCRA stage-2 threshold
38
Q_MAX = 0.95             # upper bound of speech absence probability
39
NOISE_BIAS = 1.30        # noise over-estimation factor (beta in the paper: 1.47)
40
MIN_WIN_SEC = 1.0        # minimum search window length
41
N_SUB = 8                # number of sub-windows of the minimum tracker
42
INIT_FRACTION = 0.10     # quietest frames used to initialise the noise PSD
43

44
FREQ_KERNEL = np.array([0.25, 0.5, 0.25])
45

46

47
class MinTracker:
48
    """Running minimum over ~MIN_WIN_SEC using N_SUB sub-windows."""
49

50
    def __init__(self, init, sub_len, n_sub):
51
        self.sub_len = sub_len
52
        self.buffer = np.full((max(n_sub - 1, 1), len(init)), np.inf)
53
        self.current = init.copy()
54
        self.count = 0
55

56
    def update(self, value):
57
        self.current = np.minimum(self.current, value)
58
        self.count += 1
59
        result = np.minimum(self.buffer.min(axis=0), self.current)
60
        if self.count >= self.sub_len:
61
            self.buffer = np.roll(self.buffer, -1, axis=0)
62
            self.buffer[-1] = self.current
63
            self.current = np.full_like(value, np.inf)
64
            self.count = 0
65
        return result
66

67

68
def rescale(alpha, hop_sec):
69
    return alpha ** (hop_sec / REF_HOP_SEC)
70

71

72
def enhance_channel(noisy, sample_rate):
73
    x = np.asarray(noisy, dtype=np.float64)
74
    if len(x) < 64 or not np.any(x):
75
        return x.copy()
76

77
    n_fft = int(2 ** np.round(np.log2(FRAME_SEC * sample_rate)))
78
    n_fft = max(64, min(n_fft, 1 << int(np.floor(np.log2(len(x))))))
79
    hop = n_fft // 4
80
    hop_sec = hop / sample_rate
81
    window = np.sqrt(signal.get_window("hann", n_fft))
82

83
    _, times, Y = signal.stft(
84
        x, fs=sample_rate, window=window, nperseg=n_fft,
85
        noverlap=n_fft - hop, boundary="zeros", padded=True)
86
    P = np.abs(Y) ** 2
87
    n_bins, n_frames = P.shape
88
    floor = max(P.mean() * 1e-10, 1e-30)
89

90
    # ---- initial noise PSD from the quietest complete frames ----
91
    centers = np.rint(times * sample_rate).astype(int)
92
    complete = np.flatnonzero(
93
        (centers >= n_fft // 2) & (centers + n_fft // 2 <= len(x)))
94
    if not len(complete):
95
        complete = np.arange(n_frames)
96
    energy = P[:, complete].mean(axis=0)
97
    count = max(1, int(np.ceil(INIT_FRACTION * len(complete))))
98
    chosen = complete[np.argsort(energy)[:count]]
99
    init_noise = convolve1d(P[:, chosen].mean(axis=1), FREQ_KERNEL, mode="nearest")
100
    init_noise = np.maximum(init_noise, floor)
101

102
    # ---- constants ----
103
    a_s = rescale(ALPHA_S, hop_sec)
104
    a_d = rescale(ALPHA_D, hop_sec)
105
    a_dd = rescale(ALPHA_DD, hop_sec)
106
    sub_len = max(1, int(round(MIN_WIN_SEC / hop_sec / N_SUB)))
107
    xi_min = 10 ** (XI_MIN_DB / 10)
108
    g_min = 10 ** (G_MIN_DB / 20)
109

110
    # ---- state ----
111
    lam = init_noise.copy()                 # noise PSD estimate
112
    S = init_noise.copy()                   # smoothed PSD, stage 1
113
    S_t = init_noise.copy()                 # smoothed PSD, stage 2 (speech excluded)
114
    tracker1 = MinTracker(init_noise / B_MIN, sub_len, N_SUB)
115
    tracker2 = MinTracker(init_noise / B_MIN, sub_len, N_SUB)
116
    gain = np.empty_like(P)
117
    prev_g2_gamma = None
118

119
    for l in range(n_frames):
120
        y2 = P[:, l]
121

122
        # ---------- IMCRA stage 1: coarse speech/noise indicator ----------
123
        S_f = convolve1d(y2, FREQ_KERNEL, mode="nearest")
124
        S = a_s * S + (1 - a_s) * S_f
125
        S_min = np.maximum(tracker1.update(S), floor)
126
        gamma_min = y2 / (B_MIN * S_min)
127
        zeta = S / (B_MIN * S_min)
128
        absent = ((gamma_min < GAMMA0) & (zeta < ZETA0)).astype(np.float64)
129

130
        # ---------- IMCRA stage 2: smooth only noise-like bins ----------
131
        num = convolve1d(absent * S_f, FREQ_KERNEL, mode="constant")
132
        den = convolve1d(absent, FREQ_KERNEL, mode="constant")
133
        S_new = np.where(den > 1e-9, num / np.maximum(den, 1e-9), S_t)
134
        S_t = a_s * S_t + (1 - a_s) * S_new
135
        S_t_min = np.maximum(tracker2.update(S_t), floor)
136

137
        gamma_min2 = y2 / (B_MIN * S_t_min)
138
        zeta2 = S / (B_MIN * S_t_min)
139
        q = np.where(
140
            zeta2 < ZETA0,
141
            np.clip((GAMMA1 - gamma_min2) / (GAMMA1 - 1.0), 0.0, 1.0),
142
            0.0)
143
        q = np.minimum(q, Q_MAX)
144

145
        # ---------- a priori SNR (decision-directed) ----------
146
        noise = np.maximum(NOISE_BIAS * lam, floor)
147
        gamma = y2 / noise
148
        ml = np.maximum(gamma - 1.0, 0.0)
149
        if prev_g2_gamma is None:
150
            xi = np.maximum(ml, xi_min)
151
        else:
152
            xi = np.maximum(a_dd * prev_g2_gamma + (1 - a_dd) * ml, xi_min)
153

154
        # ---------- LSA gain and speech presence probability ----------
155
        v = np.maximum(xi / (1 + xi) * gamma, 1e-8)
156
        g_h1 = np.minimum(xi / (1 + xi) * np.exp(0.5 * exp1(v)), 1.0)
157
        p = 1.0 / (1.0 + (q / (1.0 - q)) * (1.0 + xi) * np.exp(-v))
158
        gain[:, l] = g_h1 ** p * g_min ** (1.0 - p)
159
        prev_g2_gamma = g_h1 ** 2 * gamma
160

161
        # ---------- noise PSD update controlled by p ----------
162
        a_t = a_d + (1 - a_d) * p
163
        lam = np.maximum(a_t * lam + (1 - a_t) * y2, floor)
164

165
    _, enhanced = signal.istft(
166
        Y * gain, fs=sample_rate, window=window, nperseg=n_fft,
167
        noverlap=n_fft - hop, boundary=True)
168
    enhanced = enhanced[:len(x)]
169
    if len(enhanced) < len(x):
170
        enhanced = np.pad(enhanced, (0, len(x) - len(enhanced)))
171
    return np.clip(enhanced, -1, 1)
172

173

174
def process_file(source, destination):
175
    audio, sample_rate = sf.read(source, dtype="float64", always_2d=True)
176
    enhanced = np.column_stack([
177
        enhance_channel(audio[:, c], sample_rate) for c in range(audio.shape[1])])
178
    destination.parent.mkdir(parents=True, exist_ok=True)
179
    sf.write(destination, enhanced, sample_rate, subtype=sf.info(source).subtype)
180


    info = sf.info(destination)
181
    if (info.samplerate, info.channels, info.frames) != (
182
            sample_rate, audio.shape[1], audio.shape[0]):
183
        raise RuntimeError(f"Output format mismatch: {destination}")
184

185

186
def main():
187
    parser = argparse.ArgumentParser(description="IMCRA + OM-LSA denoiser")
188
    parser.add_argument("--input", type=Path, default=Path("input"))
189
    parser.add_argument("--output", type=Path, default=Path("output"))
190
    args = parser.parse_args()
191
    if args.input.resolve() == args.output.resolve():
192
        parser.error("Input and output directories must differ")
193
    if not args.input.is_dir():
194
        parser.error(f"Input directory not found: {args.input}")
195
    files = sorted(p for p in args.input.iterdir()
196
                   if p.is_file() and p.suffix.lower() == ".wav")
197
    if not files:
198
        parser.error(f"No WAV files found in {args.input}")
199
    for i, source in enumerate(files, 1):
200
        process_file(source, args.output / source.name)
201
        if i % 100 == 0 or i == len(files):
202
            print(f"Processed {i}/{len(files)}")
203

204

205
if __name__ == "__main__":
206
    main()
207
