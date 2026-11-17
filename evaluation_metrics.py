"""Shared strict evaluator: 16 kHz mono PESQ-WB and unadjusted waveform SNR.

No reference-fitted gain, external delay alignment, cropping or normalization.
PESQ performs its own standard internal processing through the installed package.
"""
import numpy as np
from pesq import pesq


def checked_pair(clean, estimate):
    clean = np.asarray(clean, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if clean.ndim != 1 or estimate.ndim != 1:
        raise ValueError('Evaluation requires mono 1-D waveforms')
    if clean.size == 0 or clean.shape != estimate.shape:
        raise ValueError('Reference and output must have identical, nonempty lengths')
    if not np.isfinite(clean).all() or not np.isfinite(estimate).all():
        raise ValueError('Reference and output must contain only finite values')
    if not np.any(clean):
        raise ValueError('Silent reference has undefined signal-to-noise ratio')
    return clean, estimate


def snr_db(clean, estimate):
    clean, estimate = checked_pair(clean, estimate)
    # Preserve the established numerical floor and per-file definition.
    return float(10*np.log10((np.sum(clean**2)+1e-12)/
                            (np.sum((clean-estimate)**2)+1e-12)))


def score_pair(clean, estimate, sample_rate):
    if sample_rate != 16000:
        raise ValueError('This evaluation protocol requires 16000 Hz PESQ-WB')
    clean, estimate = checked_pair(clean, estimate)
    return dict(pesq=float(pesq(sample_rate, clean, estimate, 'wb')),
                snr=snr_db(clean, estimate))
