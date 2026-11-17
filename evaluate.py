"""Evaluate the saved output WAVs against public validation clean references."""

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
from evaluation_metrics import score_pair

from denoise import process_file


def main():
    submission = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=submission.parent,
                        help='Directory containing validation/samples.csv')
    parser.add_argument('--output-dir', type=Path,
                        help='Evaluate already saved WAVs; otherwise run root denoise.py')
    parser.add_argument('--results', type=Path, default=submission/'validation_results.csv')
    args = parser.parse_args()
    root = args.data_root
    if not (root/'validation/samples.csv').is_file():
        parser.error('Missing validation/samples.csv; specify --data-root')
    print('Protocol: 16 kHz mono PESQ-WB; raw waveform SNR; per-file arithmetic mean')
    print('Output source:', args.output_dir if args.output_dir else submission/'denoise.py')
    with (root / "validation" / "samples.csv").open(
        newline="", encoding="utf-8-sig"
    ) as source:
        manifest = list(csv.DictReader(source))
    if not manifest:
        parser.error('No segments found in validation/samples.csv')
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
            if args.output_dir is not None:
                output_path = args.output_dir / noisy_path.name
            else:
                process_file(noisy_path, output_path)
            enhanced, output_rate = sf.read(output_path)
            if output_rate != sample_rate or enhanced.shape != clean.shape:
                raise ValueError(f"Output mismatch: {row['segment_id']}")
            noisy_scores = score_pair(clean, noisy, sample_rate)
            enhanced_scores = score_pair(clean, enhanced, sample_rate)
            results.append({
                "segment_id": row["segment_id"],
                "noise_category": row["noise_category"],
                "noisy_pesq_wb": noisy_scores['pesq'],
                "enhanced_pesq_wb": enhanced_scores['pesq'],
                "noisy_snr_db": noisy_scores['snr'],
                "enhanced_snr_db": enhanced_scores['snr'],
            })
            if index % 25 == 0 or index == len(manifest):
                print(f"Evaluated {index}/{len(manifest)}", flush=True)
    output = args.results
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    for key in ("noisy_pesq_wb", "enhanced_pesq_wb", "noisy_snr_db", "enhanced_snr_db"):
        print(f"{key}: {np.mean([row[key] for row in results]):.4f}")


if __name__ == "__main__":
    main()
