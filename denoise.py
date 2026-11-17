"""Denoise WAV files with classical raw3 signal processing."""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from raw3_dsp.processing import denoise_channel


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
