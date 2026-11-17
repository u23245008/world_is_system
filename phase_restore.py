"""Empirical original-phase interpolation at fixed pilot STFT magnitude."""
import numpy as np
from scipy import signal


def restore(noisy,pilot,sr,config):
    size=config.get('nfft',512);hop=size//4
    _,_,raw=signal.stft(noisy,fs=sr,nperseg=size,noverlap=size-hop)
    _,_,enhanced=signal.stft(pilot,fs=sr,nperseg=size,noverlap=size-hop)
    original_phase=np.exp(1j*np.angle(raw))
    pilot_phase=np.exp(1j*np.angle(enhanced))
    mix=config.get('mix',.5)
    phase=(1-mix)*pilot_phase+mix*original_phase
    phase=np.where(np.abs(phase)>1e-8,phase/np.maximum(np.abs(phase),1e-8),pilot_phase)
    phase[0]=pilot_phase[0];phase[-1]=pilot_phase[-1]
    _,output=signal.istft(np.abs(enhanced)*phase,fs=sr,nperseg=size,noverlap=size-hop)
    return np.clip(output[:len(pilot)],-1,1)
