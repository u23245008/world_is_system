"""Reference-based development evaluation; never used by denoise.py."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi
import denoise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',choices=list(denoise.METHODS),default='raw3')
    parser.add_argument('--profile', choices=['quality'], default='quality')
    parser.add_argument('--split', choices=['train', 'validation'], default='validation')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--data-root', type=Path,
                        default=Path(__file__).resolve().parent.parent,
                        help='Directory containing train/ and validation/')
    parser.add_argument('--manifest', type=Path,
                        help='Optional fixed sample CSV; audio paths are relative to data-root')
    args = parser.parse_args()
    if args.sample < 0:
        parser.error('--sample must be nonnegative')
    folder = Path(__file__).resolve().parent
    root = args.data_root.resolve()
    manifest = args.manifest if args.manifest is not None else root/args.split/'samples.csv'
    rows = list(csv.DictReader(open(manifest, encoding='utf-8-sig')))
    if 0 < args.sample < len(rows):
        groups = defaultdict(list)
        for row in rows:
            groups[row.get('noise_category', '')].append(row)
        rng = np.random.default_rng(20261116)
        for category in sorted(groups):
            rng.shuffle(groups[category])
        selected = []
        while len(selected) < args.sample and any(groups.values()):
            for category in sorted(groups):
                if groups[category] and len(selected) < args.sample:
                    selected.append(groups[category].pop())
        rows = sorted(selected, key=lambda row: row['segment_id'])
    denoise.PROFILE = args.profile
    denoise.EXTRA = denoise.BASE_EXTRA|denoise.METHODS[args.method]
    records = []
    with TemporaryDirectory() as temporary:
        for index, row in enumerate(rows, 1):
            source = root/row['noisy_path']
            clean, sr = sf.read(root/row['clean_path'])
            noisy, noisy_sr = sf.read(source)
            if sr != 16000 or noisy_sr != sr or clean.shape != noisy.shape:
                raise ValueError(f'Invalid evaluation pair: {row["segment_id"]}')
            output = Path(temporary)/source.name
            denoise.process_file(source, output)
            y, output_sr = sf.read(output)
            if output_sr != sr or y.shape != clean.shape or not np.isfinite(y).all():
                raise ValueError(f'Invalid output: {row["segment_id"]}')
            alpha = np.sum(clean*y)/max(np.sum(clean**2), 1e-12)
            target = alpha*clean
            records.append(dict(segment_id=row['segment_id'], noise_category=row.get('noise_category', ''),
                noisy_pesq_wb=pesq(sr, clean, noisy, 'wb'), enhanced_pesq_wb=pesq(sr, clean, y, 'wb'),
                noisy_snr_db=10*np.log10((np.sum(clean**2)+1e-12)/(np.sum((clean-noisy)**2)+1e-12)),
                enhanced_snr_db=10*np.log10((np.sum(clean**2)+1e-12)/(np.sum((clean-y)**2)+1e-12)),
                noisy_stoi=stoi(clean, noisy, sr, extended=False),
                enhanced_stoi=stoi(clean, y, sr, extended=False), speech_gain=alpha,
                si_sdr=10*np.log10((np.sum(target**2)+1e-12)/(np.sum((y-target)**2)+1e-12)),
                samples_at_limit=int(np.sum(np.abs(y) >= 1-1/32768))))
            if index % 20 == 0 or index == len(rows):
                print('Evaluated', index, '/', len(rows), flush=True)
    name = f'{args.split}_{args.method}'+(f'_sample{args.sample}' if args.sample else '')+'.csv'
    if args.split == 'validation' and args.method == 'raw3' and args.sample == 0:
        name = 'validation_results.csv'
    with open(folder/name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    for key in ('noisy_pesq_wb', 'enhanced_pesq_wb', 'noisy_snr_db', 'enhanced_snr_db',
                'noisy_stoi', 'enhanced_stoi', 'speech_gain', 'si_sdr'):
        print(key, round(float(np.mean([row[key] for row in records])), 5), flush=True)
    print('samples_at_limit', sum(row['samples_at_limit'] for row in records), flush=True)


if __name__ == '__main__':
    main()
