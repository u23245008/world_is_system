"""Traditional confidence-controlled harmonic postfilter.
Concept: Vondra & Vich, Adaptive Comb Filtering in Speech Enhancement (2007).
Our autocorrelation and Gaussian masks are an adapted implementation.
"""
import numpy as np
from scipy import signal
from pitch import estimate_yin
from scipy.ndimage import uniform_filter1d

def harmonic_filter(y,reference,sr,c):
 nfft=1536;hop=384
 _,times,z=signal.stft(y,fs=sr,nperseg=nfft,noverlap=nfft-hop)
 freq=np.fft.rfftfreq(nfft,1/sr)
 padded=np.pad(reference,(nfft//2,nfft//2))
 masks=np.ones(z.shape)
 for j,t in enumerate(times):
  center=int(round(t*sr));frame=padded[center:center+nfft]
  if len(frame)<nfft:frame=np.pad(frame,(0,nfft-len(frame)))
  if c.get('pitch')=='yin':
   f0,confidence=estimate_yin(frame,sr,c.get('yin_threshold',.2))
   if f0<=0 or confidence<c['threshold']:continue
  else:
   frame=frame*signal.windows.hann(nfft,sym=False)
   ac=signal.correlate(frame,frame,mode='full',method='fft')[nfft-1:]
   first=max(1,int(sr/350));last=min(nfft//2,int(sr/65))
   if first>=last:continue
   lags=np.arange(first,last)
   cum=np.r_[0,np.cumsum(frame**2)]
   norm=np.sqrt(np.maximum(cum[nfft-lags]*(cum[nfft]-cum[lags]),1e-20))
   corr=ac[lags]/norm
   peaks,_=signal.find_peaks(corr)
   if not len(peaks):continue
   peak=peaks[np.argmax(corr[peaks])];confidence=corr[peak]
   if confidence<c['threshold']:continue
   lag=lags[peak];f0=sr/lag
  distance=np.minimum(np.mod(freq,f0),f0-np.mod(freq,f0))
  comb=np.exp(-.5*(distance/c['width'])**2)
  strength=c['strength']*np.clip((confidence-c['threshold'])/(1-c['threshold']),0,1)
  mask=1-strength*(1-comb)
  mask[freq<300]=1
  masks[:,j]=mask
 if c.get('time_smooth',1)>1:
  masks=uniform_filter1d(masks,c['time_smooth'],axis=1,mode='nearest')
 z*=masks
 _,out=signal.istft(z,fs=sr,nperseg=nfft,noverlap=nfft-hop)
 return out[:len(y)]
