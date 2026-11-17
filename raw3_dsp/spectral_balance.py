"""Gentle fixed spectral equalization, empirical input-only DSP experiment."""
import numpy as np


def refine(audio,sr,c):
    pad=min(1024,len(audio)-1)
    padded=np.pad(audio,(pad,pad),mode='reflect')
    spectrum=np.fft.rfft(padded)
    frequencies=np.fft.rfftfreq(len(padded),1/sr)
    octave=np.log2(np.maximum(frequencies,.01)/c.get('center',2000))
    curve=np.exp(-.5*(octave/c.get('width',1.))**2)
    response=10**(c.get('db',1.)*curve/20)
    output=np.fft.irfft(spectrum*response,n=len(padded))[pad:pad+len(audio)]
    if c.get('normalize'):output*=np.sqrt((np.mean(audio**2)+1e-12)/(np.mean(output**2)+1e-12))
    if c.get('headroom') and np.max(np.abs(output))>.999:output*=.999/np.max(np.abs(output))
    return np.clip(output,-1,1)
