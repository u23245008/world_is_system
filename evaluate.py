"""Evaluate the saved output WAVs against public validation clean references."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
from pesq import pesq

from denoise import process_file


def snr_db(clean, estimate):
    return 10 * np.log10((np.sum(clean ** 2) + 1e-12) /
                         (np.sum((clean - estimate) ** 2) + 1e-12))


def main():
    submission = Path(__file__).resolve().parent
    root = submission.parent
    with (root / "validation" / "samples.csv").open(
        newline="", encoding="utf-8-sig"
    ) as source:
        manifest = list(csv.DictReader(source))
    results = []
    with TemporaryDirectory() as temporary:
        for index, row in enumerate(manifest, 1):
            clean, sample_rate = sf.read(root / row["clean_path"])
            noisy_path = root / row["noisy_path"]
            noisy, noisy_rate = sf.read(noisy_path)
            if sample_rate != noisy_rate or clean.shape != noisy.shape:
                raise ValueError(f"Input pair mismatch: {row['segment_id']}")
            if sample_rate != 16000:
                raise ValueError("PESQ-WB evaluation requires 16 kHz WAVs")
            output_path = Path(temporary) / noisy_path.name
            process_file(noisy_path, output_path)
            enhanced, output_rate = sf.read(output_path)
            if output_rate != sample_rate or enhanced.shape != clean.shape:
                raise ValueError(f"Output mismatch: {row['segment_id']}")
            results.append({
                "segment_id": row["segment_id"],
                "noise_category": row["noise_category"],
                "noisy_pesq_wb": pesq(sample_rate, clean, noisy, "wb"),
                "enhanced_pesq_wb": pesq(sample_rate, clean, enhanced, "wb"),
                "noisy_snr_db": snr_db(clean, noisy),
                "enhanced_snr_db": snr_db(clean, enhanced),
            })
            if index % 25 == 0 or index == len(manifest):
                print(f"Evaluated {index}/{len(manifest)}", flush=True)
    output = submission / "validation_results.csv"
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    for key in ("noisy_pesq_wb", "enhanced_pesq_wb", "noisy_snr_db", "enhanced_snr_db"):
        print(f"{key}: {np.mean([row[key] for row in results]):.4f}")


if __name__ == "__main__":
    main()
