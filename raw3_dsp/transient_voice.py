"""Transient attenuation and steady-background voice preservation for improved Raw3."""
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter1d, median_filter, uniform_filter1d

from . import raw3_next
from .tracked_tones import pitch_support

HERE = Path(__file__).resolve().parent
EPS = 1e-12


def ratio_db(a, b):
    return 10*np.log10((a+EPS)/(b+EPS))


def transient_masks(p, f, retained, voice):
    """Separate broadband attacks and lower-band bumps from sustained energy."""
    edges = (45, 260, 580, 1250, 2400, 4400, 7200)
    groups = [(f >= lo) & (f < hi) for lo,hi in zip(edges,edges[1:])]
    groups = [g for g in groups if g.any()]
    levels = np.array([p[group].sum(axis=0) for group in groups])
    trends = median_filter(levels,size=(1,39),mode="nearest")
    previous = np.pad(levels,((0,0),(3,0)),mode="edge")[:, :levels.shape[1]]
    excess = ratio_db(levels,trends)
    rise = ratio_db(levels,previous)
    impulsive = np.clip((excess-5)/12,0,1)*np.clip((rise-1.5)/8,0,1)
    impact = np.mean(impulsive,axis=0)
    coverage = np.mean((excess>6)&(rise>2),axis=0)
    impact *= np.clip((coverage-.20)/.5,0,1)
    bump = impulsive[0]*np.clip((excess[0]-9)/11,0,1)
    low_share = levels[0]/(levels.sum(axis=0)+EPS)
    bump *= np.clip((low_share-.12)/.55,0,1)
    impact = maximum_filter1d(impact, 9, mode="nearest")
    bump = maximum_filter1d(bump, 11, mode="nearest")
    p_local = median_filter(p,size=(1,29),mode="nearest")
    local_event = np.clip((ratio_db(p,p_local)-2.5)/12,0,1)
    # Speech harmonics and bins already strongly preserved by Raw3 get a
    # smaller correction; this also protects sustained syllable boundaries.
    narrow_low = np.clip((520-f)/390,0,1)
    event = np.maximum(impact[None,:],narrow_low[:,None]*bump[None,:])
    protection = (1-.9*voice)*(1-.45*retained)
    masks = {}
    for label, gate in [("strict",.26), ("broad",.13)]:
        out = np.clip((event-gate)/(1-gate),0,1)*local_event*protection
        masks[label] = gaussian_filter(out,sigma=(.75,1.4),mode="nearest")
    # A dull impact can extend across several low bands while having little
    # broadband flux. Require simultaneous short elevations in two nearby
    # bands and use a lower cutoff than the broadband impact path.
    low_groups = [(f>=lo)&(f<hi) for lo,hi in ((55,160),(160,330),(330,670))]
    lows = np.array([p[g].sum(axis=0) for g in low_groups])
    contrast = ratio_db(uniform_filter1d(lows,3,axis=1),median_filter(lows,size=(1,45),mode="nearest"))
    support = np.sort(np.clip((contrast-3.5)/10,0,1),axis=0)[-2]
    support -= .65*median_filter(support,23,mode="nearest")
    support = maximum_filter1d(np.clip(support,0,1),7,mode="nearest")
    low_taper = np.clip((885-f)/410,0,1)
    local_burst = np.clip((ratio_db(p,p_local)-1.8)/10,0,1)
    masks["low_impact"] = gaussian_filter(
        support*low_taper[:,None]*local_burst*(1-.85*voice),
        sigma=(.75,1.5),mode="nearest",
    )
    return masks


def steady_weight(p, q, full):
    """Detect a recurring quiet spectral profile without filename or noise labels."""
    energy = q[:,full].sum(axis=0)
    count = max(1,round(.26*len(full)))
    selected = full[np.argsort(energy)[:count]]
    profile = np.mean(p[:,selected],axis=1)
    # Three separated pieces must resemble the same profile for a large gate.
    unit = np.sqrt(np.maximum(profile,EPS))
    norm = np.linalg.norm(unit)
    similarities = []
    for part in np.array_split(full,3):
        if not len(part):
            continue
        n = max(1,round(.2*len(part)))
        quiet = part[np.argsort(q[:,part].sum(axis=0))[:n]]
        proxy = np.sqrt(np.maximum(p[:,quiet].mean(axis=1),EPS))
        similarities.append(np.dot(proxy,unit)/(norm*np.linalg.norm(proxy)+EPS))
    recurrence = float(np.median(similarities)) if similarities else 0.
    floor = np.maximum(uniform_filter1d(profile,3),EPS)
    strength = float(np.clip((recurrence-.64)/.29,0,1))
    return floor, strength


def prepare(noisy,sr,raw3_config=None,noise_refiner=None,noise_reference=None,raw3_output=None):
    x = np.asarray(noisy,dtype=np.float64)
    y = (raw3_next.denoise(x,sr,raw3_config,noise_refiner,noise_reference)
         if raw3_output is None else raw3_output)
    n = 2*round(.020*sr)
    hop = round(.006*sr)
    spec = dict(fs=sr,nperseg=n,noverlap=n-hop,nfft=2**int(np.ceil(np.log2(2*n))))
    f,t,X = signal.stft(x,**spec)
    _,_,Y = signal.stft(y,**spec)
    p,q = np.abs(X)**2,np.abs(Y)**2
    full = np.flatnonzero((t*sr>=n/2)&(t*sr+n/2<=len(x)))
    if not len(full):
        full = np.arange(p.shape[1])
    noise, steadiness = steady_weight(p,q,full)
    spacing = f[1]-f[0]
    pilot_bg = median_filter(q,size=(2*max(3,round(110/spacing))+1,1),mode="nearest")
    f0,reliability = pitch_support(y,sr,t,q,spacing,ratio_db(q,pilot_bg))
    cycles = f[:,None]/f0
    distance = np.abs(cycles-np.rint(cycles))*f0
    harmonic = np.clip((37-distance)/25,0,1)*(cycles>.65)
    voice = reliability*harmonic
    retained = np.clip(np.abs(Y)/(np.abs(X)+EPS),0,1)
    masks = transient_masks(p,f,retained,voice)

    local = uniform_filter1d(p,3,axis=1)
    snr = np.clip((ratio_db(local,noise[:,None])-5)/14,0,1)
    voice_band = np.clip((f-105)/100,0,1)*np.clip((2750-f)/800,0,1)
    # Narrow peaks present in the quiet noise profile are less likely to be
    # a missing speech harmonic, even if an estimated pitch runs through them.
    neighbor = median_filter(noise,size=max(5,round(230/spacing)//2*2+1),mode="nearest")
    motor_guard = 1-.8*np.clip((ratio_db(noise,neighbor)-5)/13,0,1)
    excitation = np.clip(reliability/.5,0,1)
    support = .25*excitation[None,:]+.75*voice
    restore_mask = gaussian_filter(steadiness*voice_band[:,None]*support*snr*motor_guard[:,None],
                                   sigma=(.65,1.6),mode="nearest")
    target_amp = np.sqrt(np.maximum(p-1.45*noise[:,None],0))
    missing = np.minimum(np.maximum(target_amp-np.abs(Y),0),1.05*np.abs(Y))
    boost = restore_mask*missing*(Y/(np.abs(Y)+EPS))
    residual = restore_mask*(X-Y)*np.clip(.7+.3*retained,0,1)
    return dict(base=y,spec=spec,Y=Y,transient=masks,boost=boost,residual=residual,
                steadiness=steadiness,restore_mask=restore_mask)


def apply(context,config):
    if not config:
        return context["base"].copy()
    target = context["Y"].copy()
    restore = config.get("restore",0.)
    if restore:
        target += restore*context[config.get("voice_mode","boost")]
    cut = config.get("cut_db",0.)
    if cut:
        target *= 10**(-cut*context["transient"][config["impact_gate"]]/20)
    low_cut = config.get("low_cut_db",0.)
    if low_cut:
        target *= 10**(-low_cut*context["transient"]["low_impact"]/20)
    _, y = signal.istft(target,**context["spec"])
    return np.clip(y[:len(context["base"])],-1,1-1/32768)


def denoise(audio,sr,config=None):
    x = np.asarray(audio,dtype=np.float64)
    if x.ndim!=1 or not np.isfinite(x).all() or sr<8000:
        raise ValueError("Expected finite mono audio at 8 kHz or above")
    if config is None:
        config = json.loads((HERE/"transient_voice_recommended.json").read_text())
    if not config or len(x)<max(2048,round(.08*sr)) or not np.any(x):
        return raw3_next.denoise(x,sr)
    return apply(prepare(x,sr),config)
