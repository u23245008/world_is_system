"""Traditional speech denoising; each output depends only on its noisy WAV input."""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d


NOISE_FRACTION = 0.10
NOISE_SCALE = 1.23
DD_SMOOTHING = 0.91
GAIN_FLOOR = 0.10
HIGHPASS_HZ = 122
HIGHPASS_MIX = 0.63
N_FFT = 1536


def denoise_channel(noisy, sample_rate):
    x = np.asarray(noisy, dtype=np.float64)
    if len(x) < 32 or not np.any(x):
        return x.copy()

    n_fft = min(N_FFT, len(x))
    hop = max(1, n_fft // 4)
    if HIGHPASS_HZ < sample_rate / 2:
        sos = signal.butter(3, HIGHPASS_HZ, "highpass", fs=sample_rate, output="sos")
        filtered = signal.sosfiltfilt(sos, x)
        x = (1 - HIGHPASS_MIX) * x + HIGHPASS_MIX * filtered

    _, times, spectrum = signal.stft(
        x, fs=sample_rate, window="hann", nperseg=n_fft,
        noverlap=n_fft - hop, boundary="zeros")
    power = np.abs(spectrum) ** 2
    centers = np.rint(times * sample_rate).astype(int)
    complete = np.flatnonzero(
        (centers >= n_fft // 2) & (centers + n_fft // 2 <= len(x)))
    if not len(complete):
        complete = np.arange(power.shape[1])
    frame_energy = power[:, complete].mean(axis=0)
    count = max(1, int(np.ceil(NOISE_FRACTION * len(complete))))
    selected = complete[np.argsort(frame_energy)[:count]]
    noise_power = uniform_filter1d(power[:, selected].mean(axis=1), size=3)
    noise_power = np.maximum(noise_power * NOISE_SCALE, 1e-12)

    posterior = power / noise_power[:, None]
    gain = np.empty_like(power)
    previous_clean_power = np.zeros_like(noise_power)
    for frame in range(power.shape[1]):
        instantaneous = np.maximum(posterior[:, frame] - 1, 0)
        prior = instantaneous if frame == 0 else (
            DD_SMOOTHING * previous_clean_power / noise_power
            + (1 - DD_SMOOTHING) * instantaneous)
        gain[:, frame] = np.clip(prior / (1 + prior), GAIN_FLOOR, 1)
        previous_clean_power = gain[:, frame] ** 2 * power[:, frame]

    gain = uniform_filter1d(gain, size=3, axis=0)
    gain = uniform_filter1d(gain, size=3, axis=1)
    _, enhanced = signal.istft(
        spectrum * gain, fs=sample_rate, window="hann",
        nperseg=n_fft, noverlap=n_fft - hop, boundary=True)
    return np.clip(enhanced[:len(noisy)], -1, 1)


def process_file(source, destination):
    audio, sample_rate = sf.read(source, dtype="float64", always_2d=True)
    enhanced = np.column_stack([
        denoise_channel(audio[:, channel], sample_rate)
        for channel in range(audio.shape[1])])
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, enhanced, sample_rate, subtype=sf.info(source).subtype)
    result = sf.info(destination)
    if (result.samplerate, result.channels, result.frames) != (
        sample_rate, audio.shape[1], audio.shape[0]
    ):
        raise RuntimeError(f"Output format mismatch: {destination}")


def main():
    parser = argparse.ArgumentParser(description="Denoise every WAV in input")
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output directories must differ")
    if not args.input.is_dir():
        parser.error(f"Input directory not found: {args.input}")
    files = sorted(path for path in args.input.iterdir()
                   if path.is_file() and path.suffix.lower() == ".wav")
    if not files:
        parser.error(f"No WAV files found in {args.input}")
    for index, source in enumerate(files, 1):
        process_file(source, args.output / source.name)
        if index % 100 == 0 or index == len(files):
            print(f"Processed {index}/{len(files)}")


if __name__ == "__main__":
    main()
