"""Empirical interscale wavelet shrinkage, using no training or clean input."""
import numpy as np
import pywt
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter1d


_AUDIT=None

def sure_threshold(detail,sigma):
    """Soft-threshold SURE criterion, with universal cap (Donoho/Johnstone).

    On correlated residual SWT coefficients this is an empirical adaptation,
    not a claim that the iid-Gaussian risk guarantee holds after our pipeline.
    """
    if sigma<1e-15:return 0.
    squared=np.sort((np.abs(detail)/sigma)**2)
    n=len(squared);cap=2*np.log(max(n,2))
    k=np.arange(1,n+1)
    risks=n-2*k+np.cumsum(squared)+(n-k)*squared
    valid=np.flatnonzero(squared<=cap)
    if not len(valid):return 0.
    best=valid[np.argmin(risks[valid])]
    return sigma*np.sqrt(squared[best]) if risks[best]<n else 0.


def shrink_joint(y, config):
    padded = np.pad(y, (0, (-len(y)) % 32), mode='reflect')
    wavelet = config.get('wavelet', 'sym8')
    coeffs = pywt.swt(padded, wavelet, level=5)
    out = [coeffs[0]]  # Keep the same coarse content as the four-level baseline.
    if config.get('coarse_scale',0):
        approx,detail=coeffs[0]
        blocks=detail[:len(detail)//32*32].reshape(-1,32)
        energy=np.mean(blocks**2,axis=1)
        quiet=blocks[np.argsort(energy)[:max(1,int(np.ceil(.2*len(blocks))))]]
        sigma=np.median(np.abs(quiet))/.67448975
        speech_std=np.sqrt(max(np.mean(detail**2)-sigma**2,1e-20))
        threshold=config['coarse_scale']*sigma**2/speech_std
        parent=approx*np.sqrt(np.mean(detail**2)/(np.mean(approx**2)+1e-20))
        radius=np.sqrt(detail**2+parent**2)
        coarse_gain=1/(1+(threshold/np.maximum(radius,1e-20))**2)
        coarse_gain=uniform_filter1d(coarse_gain,15,mode='reflect')
        out=[(approx,detail*coarse_gain)]
    for index in range(1, len(coeffs)):
        approx, detail = coeffs[index]
        blocks = detail[:len(detail)//32*32].reshape(-1, 32)
        energy = np.mean(blocks**2, axis=1)
        quiet = blocks[np.argsort(energy)[:max(1, int(np.ceil(.2*len(blocks))))]]
        sigma = np.median(np.abs(quiet))/.67448975
        speech_std = np.sqrt(max(np.mean(detail**2)-sigma**2, 1e-20))
        weight = config.get('scale_weights', (1., 1., 1., 1.))[index-1]
        threshold = config['scale']*weight*sigma**2/speech_std
        if threshold <= 1e-15:
            shrunk = detail.copy()
        elif config['method'] == 'envelope':
            envelope_power = np.abs(hilbert(detail))**2
            radius = np.sqrt(np.maximum(uniform_filter1d(
                envelope_power, size=config.get('window', 1), mode='reflect'), 1e-20))
            shrink_gain = np.maximum(1-threshold/radius, 0)
            if config.get('gain_smooth', 1) > 1:
                shrink_gain = uniform_filter1d(shrink_gain,
                    size=config['gain_smooth'], mode='reflect')
            shrunk = detail*shrink_gain
        elif config['method'] in ('bivariate', 'rational'):
            parent = coeffs[index-1][1]
            if config.get('align', False):
                parent = np.roll(parent, _relative_delay(wavelet, index))
            ratio = np.sqrt(np.mean(detail**2)/(np.mean(parent**2)+1e-20))
            radius = np.sqrt(detail**2+config['parent']*(parent*ratio)**2)
            if config.get('radius_smooth', 1) > 1:
                radius = np.sqrt(np.maximum(uniform_filter1d(radius**2,
                    size=config['radius_smooth'], mode='reflect'), 1e-20))
            if config['method'] == 'bivariate':
                shrink_gain = np.maximum(1-threshold/np.maximum(radius, 1e-20), 0)
            else:
                shrink_gain = 1/(1+(threshold/np.maximum(radius, 1e-20))**config.get('power', 2))
            if config.get('gain_smooth', 1) > 1:
                shrink_gain = uniform_filter1d(shrink_gain,
                    size=config['gain_smooth'], mode='reflect')
            shrunk = detail*shrink_gain
        elif config['method'] == 'firm':
            shrunk = pywt.threshold_firm(detail, threshold, config['ratio']*threshold)
        else:
            shrunk = pywt.threshold(detail, threshold, mode=config['method'])
        if config.get('risk_mix',0):
            threshold_sure=sure_threshold(detail,sigma)
            alternative=pywt.threshold(detail,threshold_sure,mode='soft')
            mix=config['risk_mix']
            shrunk=(1-mix)*shrunk+mix*alternative
        out.append((approx, shrunk))
    filtered = pywt.iswt(out, wavelet)[:len(y)]
    power = np.mean(filtered**2)+1e-12
    gain = np.clip(np.sqrt((.5*power+.5*np.mean(y**2))/power), 1, 1.2)
    if _AUDIT is not None:_AUDIT.append(dict(stage='wavelet_restore',peak=float(np.max(np.abs(filtered*gain))),samples_above_limit=int(np.sum(np.abs(filtered*gain)>1))))
    return np.clip(filtered*gain, -1, 1)


_delays = {}


def _relative_delay(wavelet, index):
    if wavelet not in _delays:
        impulse = np.zeros(4096)
        impulse[2048] = 1
        coefficients = pywt.swt(impulse, wavelet, level=5)
        sample = np.arange(len(impulse))
        centers = [np.sum(sample*d*d)/max(np.sum(d*d), 1e-20)
                   for _, d in coefficients]
        _delays[wavelet] = [int(round(centers[i]-centers[i-1]))
                           for i in range(1, len(centers))]
    return _delays[wavelet][index-1]
