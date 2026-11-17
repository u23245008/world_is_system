"""Conservative stationary-line detection and sinusoidal projection subtraction.

Own offline DSP variant; related concept: periodic interference cancellation in
Widrow et al. (1975). Uses no separate noise recording, clean speech or learning.
"""
import numpy as np
from scipy import signal,fft
from scipy.ndimage import uniform_filter1d


def remove(x,sr,c):
    if len(x)<sr:return x.copy()
    size=fft.next_fast_len(4*len(x))
    power=np.abs(np.fft.rfft(x*signal.windows.hann(len(x),sym=False),n=size))**2
    resolution=sr/size
    logarithm=np.log(np.maximum(power,1e-20))
    background=uniform_filter1d(logarithm,max(3,int(20/resolution)|1))
    contrast=(logarithm-background)*10/np.log(10)
    peaks,_=signal.find_peaks(contrast,height=c.get('contrast',16),prominence=6,distance=max(1,int(8/resolution)))
    peaks=[p for p in peaks if 80<p*resolution<min(6000,sr*.45)]
    peaks=sorted(peaks,key=lambda p:-power[p])[:c.get('max_lines',12)]
    t=np.arange(len(x))/sr
    out=x.copy();blocks=np.array_split(np.arange(len(x)),12)
    for peak in peaks:
        a,b,d=logarithm[peak-1:peak+2]
        delta=np.clip(.5*(a-d)/(a-2*b+d),-.5,.5) if abs(a-2*b+d)>1e-12 else 0
        frequency=(peak+delta)*resolution
        angle=2*np.pi*frequency*t
        carrier=np.exp(-1j*angle)
        coefficients=np.array([2*np.mean(x[ids]*carrier[ids]) for ids in blocks])
        coherence=abs(np.mean(coefficients))**2/max(np.mean(abs(coefficients)**2),1e-20)
        persistence=np.percentile(abs(coefficients),20)/max(np.percentile(abs(coefficients),80),1e-12)
        stationary=coherence>=c.get('coherence',.85) and persistence>=.4
        if not stationary:
            if not c.get('track_drift') or persistence<c.get('drift_persistence',.6):continue
            centers=np.array([np.mean(t[ids]) for ids in blocks])-np.mean(t)
            phases=np.unwrap(np.angle(coefficients))
            model=np.polynomial.polynomial.polyfit(centers,phases,2,w=np.sqrt(np.maximum(abs(coefficients),1e-12)))
            fitted=np.polynomial.polynomial.polyval(centers,model)
            consistency=abs(np.sum(abs(coefficients)*np.exp(1j*(phases-fitted))))/max(np.sum(abs(coefficients)),1e-12)
            offset=model[1]/(2*np.pi);drift=model[2]/np.pi
            if consistency<c.get('drift_coherence',.98) or abs(drift)>c.get('max_drift',1.) or abs(offset)>2:continue
            angle=angle+np.polynomial.polynomial.polyval(t-np.mean(t),model)
        design=np.column_stack((np.cos(angle),np.sin(angle)))
        parameters=np.linalg.solve(design.T@design,design.T@out)
        estimate=design@parameters
        if c.get('amplitude_tracking'):
            carrier=np.exp(-1j*angle)
            coefficients=np.array([2*np.mean(x[ids]*carrier[ids]) for ids in blocks])
            smoothed=uniform_filter1d(coefficients.real,c.get('amplitude_window',5))+1j*uniform_filter1d(coefficients.imag,c.get('amplitude_window',5))
            centers=np.array([np.mean(t[ids]) for ids in blocks])
            dynamic=np.interp(t,centers,smoothed.real)+1j*np.interp(t,centers,smoothed.imag)
            constant=parameters[0]-1j*parameters[1]
            delta=dynamic-constant
            bound=.75*abs(constant)
            delta*=np.minimum(1,bound/np.maximum(abs(delta),1e-20))
            estimate+=c.get('amplitude_mix',.5)*np.real(delta*np.exp(1j*angle))
        out-=c.get('strength',.75)*estimate
    return out
