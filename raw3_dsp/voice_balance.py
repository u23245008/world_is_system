"""Own input-only, speech-confidence-gated restoration of voice body.

This is a smooth limited spectral boost, not a trained speech classifier.
Noise statistics come from the processed input alone; no clean references.
"""
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


def refine(pilot,sr,c,noisy=None):
    if c.get('db',0)==0:return pilot.copy()
    size=c.get('nfft',1024);hop=size//4
    frequencies,times,z=signal.stft(pilot,fs=sr,nperseg=size,noverlap=size-hop)
    power=abs(z)**2
    if c.get('raw_gate'):
        if noisy is None:raise ValueError('raw_gate requires the same noisy input')
        _,_,raw=signal.stft(noisy,fs=sr,nperseg=size,noverlap=size-hop)
        power=abs(raw)**2
    complete=np.flatnonzero((times*sr>=size//2)&(times*sr+size//2<=len(pilot)))
    selected=complete[np.argsort(power[:,complete].mean(0))[:max(1,int(np.ceil(.1*len(complete))))]]
    noise=np.maximum(uniform_filter1d(np.median(power[:,selected],axis=1)/np.log(2),5)*c.get('noise_factor',1.5),1e-14)
    local=uniform_filter1d(uniform_filter1d(power,3,axis=0),3,axis=1)
    ratio_db=10*np.log10(np.maximum(local/noise[:,None],1e-8))
    gate=np.clip((ratio_db-c.get('threshold',3))/9,0,1)
    gate=uniform_filter1d(uniform_filter1d(gate,5,axis=0),5,axis=1)
    if c.get('phase_gate'):
        unit=np.exp(1j*np.angle(z));coherence=np.ones(z.shape)
        coherence[:,1:-1]=.5+.5*np.real(unit[:,2:]*unit[:,:-2]*np.conj(unit[:,1:-1])**2)
        coherence=uniform_filter1d(uniform_filter1d(coherence,3,axis=0),3,axis=1)
        gate*=np.clip((coherence-.75)/.2,0,1)
    bell=np.exp(-.5*(np.log2(np.maximum(frequencies,.01)/c.get('center',500))/c.get('width',1.))**2)
    gain=10**(bell[:,None]*(c['db']*gate-c.get('noise_cut_db',0)*(1-gate))/20)
    _,out=signal.istft(z*gain,fs=sr,nperseg=size,noverlap=size-hop)
    out=out[:len(pilot)]
    if c.get('normalize'):
        out*=np.sqrt((np.mean(pilot**2)+1e-12)/(np.mean(out**2)+1e-12))
    peak=np.max(np.abs(out))
    if peak>.999:out*=.999/peak
    return out
