"""Traditional fixed-prior SPP tracking and onset-preserving gain smoothing.
No learned models; noisy input alone.
"""
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d, maximum_filter1d, median_filter
from scipy.special import exp1

def enhance(x,sr,c):
    nfft=c['nfft']; hop=nfft//c.get('hop_divisor',4)
    original=x
    reference=c.get('noise_reference')
    if c['hp']:
        hp=signal.butter(3,c['hp'],'highpass',fs=sr,output='sos')
        x=signal.sosfiltfilt(hp,x)
        if reference is not None:
            reference=signal.sosfiltfilt(hp,reference)
    kind=c.get('analysis_window','hann')
    if kind=='hann_power':window=signal.get_window('hann',nfft)**c.get('window_power',.5)
    elif kind=='sqrt_hann':window=np.sqrt(signal.get_window('hann',nfft))
    else:window=signal.get_window(kind,nfft)
    f,t,z=signal.stft(x,fs=sr,window=window,nperseg=nfft,noverlap=nfft-hop)
    power=abs(z)**2
    pitch_q=None
    complete=np.flatnonzero((t*sr>=nfft//2)&(t*sr+nfft//2<=len(x)))
    proposal=None
    if c.get('noise_refiner') is not None:
        reference_spec=None if reference is None else signal.stft(reference,fs=sr,window=window,
                                                                  nperseg=nfft,noverlap=nfft-hop)[2]
        proposal=c['noise_refiner'](f,t,power,complete,reference_spec)
    selected=complete[np.argsort(power[:,complete].mean(0))[:max(1,int(np.ceil(c.get('fraction',.1)*len(complete))))]]
    initial_estimate=power[:,selected].mean(1)
    if c.get('noise_init')=='median':
        # Approximate exponential-periodogram median calibration, not a learned estimate.
        initial_estimate=np.median(power[:,selected],axis=1)/np.log(2)
    initial=np.maximum(uniform_filter1d(initial_estimate,c.get('noise_smooth',3)),1e-12)
    if c.get('quantile',0):
        smooth=uniform_filter1d(power[:,complete],size=5,axis=1)
        profile=uniform_filter1d(np.percentile(smooth,c['quantile'],axis=1),size=3)*c.get('quantile_scale',1.)
        initial=(1-c.get('quantile_mix',1.))*initial+c.get('quantile_mix',1.)*profile
        initial=np.maximum(initial,1e-12)
    prior_power=uniform_filter1d(power,c.get('prior_time',1),axis=1)
    noise=initial.copy(); prev=np.zeros_like(noise); probavg=np.zeros_like(noise)
    tracker=None
    gains=[]
    for j in range(z.shape[1]):
        p=power[:,j]
        if tracker is not None:
            imcra_q=tracker.absence(p)
            prob=1-imcra_q
        elif c['track']:
            gamma=p/np.maximum(noise,1e-12)
            if c.get('detection_smooth',1)>1:
                width=c['detection_smooth']
                gamma=uniform_filter1d(p,width)/np.maximum(uniform_filter1d(noise,width),1e-12)
            prior=c.get('prior',15.)
            if c.get('flatness_prior'):
                flatness=np.exp(np.mean(np.log(np.maximum(gamma,1e-12))))/max(np.mean(gamma),1e-12)
                prior=5+45*np.clip((flatness-.15)/.55,0,1)
            prob=1/(1+(1+prior)*np.exp(-np.minimum(gamma*prior/(1+prior),700)))
            if 'probavg' in c:probavg=c['probavg']*probavg+(1-c['probavg'])*prob
            else:probavg=.9*probavg+.1*prob
            prob=np.where(probavg>.99,np.minimum(prob,.99),prob)
            estimate=prob*noise+(1-prob)*p
            # Fixed-prior SPP: Gerkmann & Hendriks (2012), DOI 10.1109/TASL.2011.2180896.
            # Global low-energy initialization and lower bound are our adaptations.
            rate=c.get('rate',.8)
            noise=rate*noise+(1-rate)*estimate
            if c.get('noise_reg',0):
                # Empirical local regularization of noise PSD in noise-like bins.
                log_noise=np.log(np.maximum(noise,1e-12))
                weight=c['noise_reg']*(1-prob)
                noise=np.exp(log_noise+weight*(uniform_filter1d(log_noise,5)-log_noise))
            noise=np.maximum(noise,c.get('noise_bound',.1)*initial)
        scale=c['scale']
        if c.get('adaptive_scale'):
            speech_scale, noise_scale=c['adaptive_scale']
            # Adapt using only noisy-input SPP; no learned classifier or clean mask.
            presence=uniform_filter1d(prob,5)
            scale=noise_scale+(speech_scale-noise_scale)*presence
        for_gain=noise if proposal is None else proposal(noise,j)
        scaled_noise=np.maximum(for_gain*scale,1e-12)
        gamma=p/scaled_noise
        xi=np.maximum(prior_power[:,j]/scaled_noise-1,0) if j==0 else c.get('dd',.95)*prev/scaled_noise+(1-c.get('dd',.95))*np.maximum(prior_power[:,j]/scaled_noise-1,0)
        w=xi/(1+xi)
        if c['est']=='lsa': g=w*np.exp(.5*exp1(np.maximum(w*gamma,1e-8)))
        elif c['est']=='sub': g=np.sqrt(np.maximum(1-c['scale']/np.maximum(gamma,1e-12),0))
        elif c['est']=='omlsa':
            v=np.maximum(w*gamma,1e-8)
            gh=np.clip(w*np.exp(.5*exp1(v)),1e-8,1)
            q=c.get('q',.5)
            if pitch_q is not None:q=pitch_q[:,j]
            if tracker is not None:
                blend=c.get('imcra_gain',0)
                q=(1-blend)*q+blend*imcra_q
            sp=1/(1+q/(1-q)*(1+xi)*np.exp(-v))
            g=gh**sp*c['floor']**(1-sp)
        elif c['est']=='presence_wiener':
            v=np.maximum(w*gamma,1e-8)
            q=c.get('q',.5)
            sp=1/(1+q/(1-q)*(1+xi)*np.exp(-v))
            g=w**sp*c['floor']**(1-sp)
        elif c['est']=='subpower' :g=np.sqrt(np.maximum(1-1/np.maximum(gamma,1e-12),0))
        elif c['est']=='subamp':g=np.maximum(1-1/np.sqrt(np.maximum(gamma,1e-12)),0)
        else:g=w
        g=np.clip(g,c['floor'],1)**c.get('power',1.)
        prev=g*g*p
        if tracker is not None:
            noise=np.maximum(tracker.update(noise,p,xi,gamma,imcra_q),.1*initial)
        if c.get('reliable_mix',0):
            # RFSNR-inspired soft reliability selection. DD feedback above remains unchanged.
            # Plapous, Marro, Scalart, EUSIPCO 2006, equations (14) and step 4.
            # Our sigmoid gates and mixture with OM-LSA are adaptations, not a replica.
            eta=10**(-6/10)
            reliability=(1/(1+np.exp(-np.clip(4*np.log(np.maximum(xi,1e-12)/eta),-50,50))))
            reliability*=1/(1+np.exp(-np.clip(4*np.log(np.maximum(gamma,1e-12)/eta),-50,50)))
            direct=np.maximum(1-1/np.maximum(gamma,1e-12),0)
            selected=reliability*direct+(1-reliability)*c['floor']
            g=(1-c['reliable_mix'])*g+c['reliable_mix']*selected
        gains.append(g)
    gain=np.array(gains).T
    if c.get('reverse',False):
        reverse_c=c|{'reverse':False}
        if c.get('noise_reference') is not None:
            reverse_c['noise_reference']=c['noise_reference'][::-1].copy()
        # Symmetric offline processing uses only the same noisy input.
        rev=enhance(original[::-1].copy(),sr,reverse_c)[::-1]
    else: rev=None
    gain=uniform_filter1d(uniform_filter1d(gain,c.get('fsmooth',3),axis=0),c.get('tsmooth',3),axis=1)
    if c.get('onset',0):
        gain=(1-c['onset'])*gain+c['onset']*maximum_filter1d(gain,c.get('onset_size',3),axis=1)
    if c.get('median',0):
        gain=median_filter(gain,size=(c['median'],c['median']))
    if c.get('shape_gain'):
        # Fixed smooth odds remapping: reduce uncertain gains, preserve strong speech.
        shape=c['shape_gain'];bounded=np.clip(gain,1e-8,1-1e-8)
        odds=shape.get('power',1.5)*(np.log(bounded)-np.log1p(-bounded))-np.log(shape.get('penalty',1.))
        gain=1/(1+np.exp(np.clip(-odds,-700,700)))
    if c.get('island', 0):
        # Empirical suppression of isolated low-confidence gain peaks.
        # Strong gains and sustained/time-frequency-supported components are protected.
        log_gain = np.log(np.maximum(gain, c['floor']))
        neighborhood = median_filter(log_gain, size=(3, 5), mode='reflect')
        if c.get('island_mode') == 'cross':
            neighborhood = np.maximum(
                median_filter(log_gain, size=(1,c.get('island_time',5)), mode='reflect'),
                median_filter(log_gain, size=(5,1), mode='reflect'))
        excess = np.maximum(log_gain-neighborhood, 0)
        uncertainty = np.clip(1-gain/.65, 0, 1)**2
        if c.get('island_mode') == 'cross':
            uncertainty=.25+.75*uncertainty
        gain *= np.exp(-c['island']*uncertainty*excess)
    if c.get('cep',0):
        logg=np.log(np.maximum(gain,1e-8))
        cep=np.fft.irfft(logg,n=nfft,axis=0)
        # Smooth short-quefrency envelope less than high-quefrency fluctuations.
        q=np.minimum(np.arange(nfft),nfft-np.arange(nfft))/sr
        rates=np.where((q<.001)|((q>.003)&(q<.015)),.3,c['cep'])
        state=cep[:,0].copy()
        for j in range(cep.shape[1]):
            state=rates*state+(1-rates)*cep[:,j]
            cep[:,j]=state
        gain=np.clip(np.exp(np.fft.rfft(cep,axis=0).real),c['floor'],1)
    if c.get('adaptive_band',0):
        freq=np.fft.rfftfreq(nfft,1/sr)
        band=np.abs(signal.sosfreqz(signal.butter(3,c['lp'],'lowpass',fs=sr,output='sos'),worN=freq,fs=sr)[1])**2
        reliable=uniform_filter1d(gain,5,axis=1)**c['adaptive_band']
        gain*=band[:,None]+(1-band[:,None])*(c.get('highmix',.05)+(1-c.get('highmix',.05))*reliable)
    _,y=signal.istft(z*gain,fs=sr,window=window,nperseg=nfft,noverlap=nfft-hop)
    if c.get('phase_iters',0):
        target=np.abs(z*gain)
        for _ in range(c['phase_iters']):
            _,_,consistent=signal.stft(y[:len(original)],fs=sr,window=window,nperseg=nfft,noverlap=nfft-hop)
            updated=target*np.exp(1j*np.angle(consistent))
            _,y=signal.istft(updated,fs=sr,window=window,nperseg=nfft,noverlap=nfft-hop)
    y=y[:len(original)]
    if c['lp'] and not c.get('adaptive_band',0):
        low=signal.sosfiltfilt(signal.butter(3,c['lp'],'lowpass',fs=sr,output='sos'),y)
        y=c.get('highmix',0)*y+(1-c.get('highmix',0))*low
    mix=c.get('mix',.015)
    y=(1-mix)*y+mix*original
    if rev is not None:y=.5*(y+rev)
    return np.clip(y,-1,1)
