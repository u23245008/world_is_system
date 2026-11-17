"""YIN-inspired deterministic F0 estimator; no learned model.

Difference, cumulative normalization, first acceptable dip and raw-difference
parabolic interpolation follow de Cheveigne/Kawahara (2002), steps 2--5.
The original paper's best-local-estimate search is not implemented.
"""
import numpy as np
from scipy import signal


def estimate_yin(frame,sr,threshold=.2):
    x=np.asarray(frame,dtype=float)-np.mean(frame)
    if np.mean(x*x)<1e-18:return 0.,0.
    last=min(len(x)//2-1,int(sr/65))
    first=max(2,int(sr/350))
    if first>=last:return 0.,0.
    size=len(x)-last
    head=x[:size]
    cross=signal.correlate(x,head,mode='valid',method='fft')
    cumulative=np.r_[0,np.cumsum(x*x)]
    energy=cumulative[size:]-cumulative[:-size]
    difference=np.maximum(np.sum(head*head)+energy-2*cross,0)
    normalized=np.ones(last+1)
    normalized[1:]=difference[1:]*np.arange(1,last+1)/np.maximum(np.cumsum(difference[1:]),1e-20)
    minima,_=signal.find_peaks(-normalized)
    minima=minima[(minima>=first)&(minima<last)]
    candidates=minima[normalized[minima]<threshold]
    lag=int(candidates[0]) if len(candidates) else first+int(np.argmin(normalized[first:last]))
    confidence=float(np.clip(1-normalized[lag],0,1))
    offset=0.
    if 0<lag<last:
        denominator=difference[lag-1]-2*difference[lag]+difference[lag+1]
        if abs(denominator)>1e-20:
            offset=float(np.clip(.5*(difference[lag-1]-difference[lag+1])/denominator,-.5,.5))
    return sr/(lag+offset),confidence
