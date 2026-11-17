"""Gentle noise-only energy expander; no clean reference or learned model."""
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, gaussian_filter1d


def filter_pauses(y, sr, floor=.7, threshold=6, hold=.12):
    nfft=512
    hop=128
    _,_,z=signal.stft(y,fs=sr,nperseg=nfft,noverlap=nfft-hop)
    frequencies=np.fft.rfftfreq(nfft,1/sr)
    p=np.mean(np.abs(z[(frequencies>=150)&(frequencies<=min(6000,sr*.45))])**2,axis=0)
    # Protect neighboring consonants and word endings; no hard zero/gating.
    noise=max(np.percentile(p,15),1e-16)
    db=10*np.log10(np.maximum(p,1e-16)/noise)
    presence=1/(1+np.exp(-np.clip((db-threshold)/max(1,threshold/3),-50,50)))
    presence=maximum_filter1d(presence,size=max(3,int(hold*sr/hop)//2*2+1),mode='nearest')
    presence=gaussian_filter1d(presence,sigma=max(1,.015*sr/hop),mode='nearest')
    gain=floor+(1-floor)*presence
    _,result=signal.istft(z*gain,fs=sr,nperseg=nfft,noverlap=nfft-hop)
    return result[:len(y)]
