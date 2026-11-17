"""Confidence-controlled low-frequency restoration from the noisy-only pilot."""
import numpy as np
from scipy import signal
from .pitch import estimate_yin
from scipy.ndimage import uniform_filter1d


def restore_bass(main, low, correction, sr, config):
    nfft, hop = 1536, 384
    frequencies, _, z = signal.stft(correction, fs=sr, nperseg=nfft,
                                    noverlap=nfft-hop)
    padded = np.pad(low, (nfft//2, nfft//2))
    window = signal.windows.hann(nfft, sym=False)
    lags = np.arange(max(1, int(sr/350)), min(nfft//2, int(sr/65)))
    base=config.get("base",.5)
    mask = np.full(z.shape, base)
    for frame in range(z.shape[1]):
        segment = padded[frame*hop:frame*hop+nfft]
        if len(segment) < nfft:
            segment = np.pad(segment, (0, nfft-len(segment)))
        if config.get('pitch')=='yin':
            f0, raw_confidence = estimate_yin(segment,sr,config.get('yin_threshold',.2))
            confidence=np.clip((raw_confidence-config['confidence'])/(1-config['confidence']),0,1)
            if f0<=0 or confidence==0:continue
        else:
            segment = segment*window
            if not len(lags):
                continue
            ac = signal.correlate(segment, segment, mode='full', method='fft')[nfft-1:]
            cumulative = np.r_[0, np.cumsum(segment**2)]
            norm = np.sqrt(np.maximum(cumulative[nfft-lags]*
                            (cumulative[nfft]-cumulative[lags]), 1e-20))
            correlation = ac[lags]/norm
            peaks, _ = signal.find_peaks(correlation)
            if not len(peaks):
                continue
            peak = peaks[np.argmax(correlation[peaks])]
            confidence = np.clip((correlation[peak]-config['confidence'])/
                                 (1-config['confidence']), 0, 1)
            if confidence == 0:
                continue
            f0 = sr/lags[peak]
        distance = np.minimum(np.mod(frequencies, f0), f0-np.mod(frequencies, f0))
        harmonic = np.exp(-.5*(distance/config['width'])**2)
        harmonic[frequencies < 50] = 0
        target = config['background']+(config['peak']-config['background'])*harmonic
        mask[:, frame] = base+config.get("mix",1.)*confidence*(target-base)
    mask = uniform_filter1d(mask, 3, axis=1, mode='nearest')
    _, adaptive = signal.istft(z*mask, fs=sr, nperseg=nfft, noverlap=nfft-hop)
    return main+adaptive[:len(main)]
