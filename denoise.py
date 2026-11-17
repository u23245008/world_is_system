"""Traditional speech enhancement with smooth shrinkage and voice-protected bass restoration.

The enhancement uses each noisy input alone. Clean references are never read.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d


from coefficient_filter import shrink_joint
from spp import enhance
from harmonic import harmonic_filter
from omlsa import enhance_channel

CONFIG = dict(nfft=1536,hp=170,lp=2500,track=True,est='dd',scale=1.,floor=.05,dd=.92,highmix=.05,mix=.015,onset=.5)
PROFILE = 'quality'
_AUDIT=None
BASE_EXTRA = dict(highmix=.15,bass=dict(base=.4,mix=.5,background=.1,peak=.8,width=30,confidence=.5),wave=dict(gain_smooth=7),pause_floor=.5,pause_threshold=2,pause_hold=.2)

COMMON = dict(island=2.,island_mode='cross',pause_floor=.15,pause_threshold=5,wave=dict(gain_smooth=7,scale=40,coarse_scale=8),noise_init='median',dd=.75,bass_pitch='yin',harmonic_pitch='yin',mix=0.,phase_restore=dict(nfft=1024,mix=.5),reverse=True)
METHODS = {'balanced':COMMON,'maxscore':COMMON|dict(bass=dict(base=.2,mix=.5,background=.1,peak=.8,width=30,confidence=.5))}
METHODS['cycle4'] = COMMON|dict(cycle_shifts=(0,128,256,384))
METHODS['previous']=METHODS['cycle4']
METHODS['snr']=METHODS['cycle4']|dict(level_mix=.75,wave=COMMON['wave']|dict(parent=.5),tonal_cancel=dict(coherence=.65,strength=1.,contrast=16))
METHODS['joint']=METHODS['snr']|dict(spectral_balance=dict(center=500,db=.5,width=1.))
METHODS['gentle']=METHODS['snr']|dict(spectral_balance=dict(center=500,db=.25,width=1.))
METHODS={key:METHODS[key] for key in ('previous','snr','gentle','joint')}
METHODS={'gentle':METHODS['gentle']}
METHODS['flat40']=METHODS['gentle']|dict(trial_post=dict(kind='flat',db=1.,center=500,normalize=True,headroom=True))
METHODS['voice40']=METHODS['gentle']|dict(trial_post=dict(kind='voice',db=1.25,center=500,normalize=True))
METHODS['phase40']=METHODS['gentle']|dict(trial_post=dict(kind='voice',db=3.,center=500,normalize=True,phase_gate=True,threshold=9))
METHODS['raw2']=METHODS['gentle']|dict(trial_post=dict(kind='voice',db=2.,threshold=0,center=500,normalize=True,raw_gate=True))
METHODS['raw3']=METHODS['gentle']|dict(trial_post=dict(kind='voice',db=3.,threshold=0,center=500,normalize=True,raw_gate=True))
EXTRA = BASE_EXTRA|METHODS['raw3']

def _single_channel(noisy, sample_rate):
    x = np.asarray(noisy, dtype=np.float64)
    if len(x)<2048 or not np.any(x): return x.copy()
    config=CONFIG|dict(hp=min(170,sample_rate*.1),lp=min(2250,sample_rate*.45),floor=.01,scale=1.3,dd=.8,highmix=.1,rate=.7)
    config=config|dict(est='omlsa',q=.1)
    low=enhance(x,sample_rate,CONFIG|dict(hp=0,lp=0,nfft=2048,highmix=0,mix=0)|EXTRA.get('low_config',{}))
    config=config|EXTRA
    if EXTRA.get('pitch_q'):config=config|dict(pitch_reference=low)
    main=enhance(x,sample_rate,config)
    correction=signal.sosfiltfilt(signal.butter(3,min(250,sample_rate*.2),'lowpass',fs=sample_rate,output='sos'),low-main)
    amount={'quality':.5}[PROFILE]
    y=main+amount*correction
    if EXTRA.get('bass'):
        from adaptive_bass import restore_bass
        y=restore_bass(main,low,correction,sample_rate,EXTRA['bass']|dict(pitch=EXTRA.get('bass_pitch','ac'),yin_threshold=EXTRA.get('yin_threshold',.2)))
    y=harmonic_filter(y,low,sample_rate,dict(strength=EXTRA.get('harmonic_strength',.3),width=30,threshold=.4,time_smooth=EXTRA.get('harmonic_time',1),pitch=EXTRA.get('harmonic_pitch','ac'),yin_threshold=EXTRA.get('yin_threshold',.2)))
    # Noisy-only loudness reference. Clean is never read here.
    reference=enhance_channel(x,sample_rate)
    power=np.mean(y*y)+1e-12
    level_mix=EXTRA.get('level_mix',.5)
    target=(1-level_mix)*power+level_mix*np.mean(reference**2)
    gain=np.clip(np.sqrt(target/power),1.,EXTRA.get('level_cap',1.5))
    if _AUDIT is not None:_AUDIT.append(dict(stage='level_restore',peak=float(np.max(np.abs(y*gain))),samples_above_limit=int(np.sum(np.abs(y*gain)>1))))
    output=np.clip(y*gain,-1,1)
    output=shrink_joint(output,dict(method='rational',wavelet='bior4.4',parent=1.,power=2,scale=48,align=True,scale_weights=(1.25,1.25,.75,.5))|EXTRA.get('wave',{}))
    if EXTRA.get('pause_floor', 1) < 1:
        from pause_filter import filter_pauses
        output=filter_pauses(output,sample_rate,EXTRA['pause_floor'],
            threshold=EXTRA.get('pause_threshold',6), hold=EXTRA.get('pause_hold',.12))
    if EXTRA.get('bm_post'):
        from bm_post import refine
        output=refine(output,sample_rate,EXTRA['bm_post'])
    if EXTRA.get('harmonic_restore'):
        from harmonic_restore import restore
        output=restore(x,output,low,sample_rate,EXTRA['harmonic_restore'])
    if EXTRA.get('phase_restore'):
        from phase_restore import restore
        output=restore(x,output,sample_rate,EXTRA['phase_restore'])
    if EXTRA.get('envelope_post'):
        from envelope_post import refine
        output=refine(output,sample_rate,EXTRA['envelope_post'])
    return output


def _cycle_channel(noisy,sample_rate):
    x=np.asarray(noisy,dtype=np.float64)
    shifts=EXTRA.get('cycle_shifts',(0,))
    first=_single_channel(x,sample_rate)
    if len(shifts)==1 or len(x)<2048 or not np.any(x):return first
    others=[]
    for offset in shifts[1:]:
        padded=np.pad(x,(offset,offset),mode='reflect')
        others.append(_single_channel(padded,sample_rate)[offset:offset+len(x)])
    return (first+sum(others))/len(shifts)


def denoise_channel(noisy,sample_rate):
    x=np.asarray(noisy,dtype=np.float64)
    alpha=EXTRA.get('preemphasis',0.)
    if len(x)<2048 or not np.any(x):return x.copy()
    if EXTRA.get('tonal_cancel'):
        from tonal_cancel import remove
        x=remove(x,sample_rate,EXTRA['tonal_cancel'])
    if not alpha:output=_cycle_channel(x,sample_rate)
    else:
        emphasized=signal.lfilter([1.,-alpha],[1.],x)
        filtered=_cycle_channel(emphasized,sample_rate)
        output=np.clip(signal.lfilter([1.],[1.,-alpha],filtered),-1,1)
    if EXTRA.get('spectral_balance'):
        from spectral_balance import refine
        output=refine(output,sample_rate,EXTRA['spectral_balance'])
    if EXTRA.get('trial_post'):
        c=EXTRA['trial_post']
        if c.get('kind')=='flat':
            from spectral_balance import refine
            output=refine(output,sample_rate,c)
        else:
            from voice_balance import refine
            output=refine(output,sample_rate,c,noisy=noisy)
    return output


def process_file(source, destination):
    audio, sample_rate = sf.read(source, dtype='float64', always_2d=True)
    output = np.column_stack([
        denoise_channel(audio[:, channel], sample_rate)
        for channel in range(audio.shape[1])])
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, output, sample_rate, subtype=sf.info(source).subtype)
    info = sf.info(destination)
    if (info.samplerate, info.channels, info.frames) != (
            sample_rate, audio.shape[1], audio.shape[0]):
        raise RuntimeError(f'Output format mismatch: {destination}')


def main():
    parser = argparse.ArgumentParser(description='Denoise every WAV in input')
    parser.add_argument('--method',choices=list(METHODS),default='raw3')
    parser.add_argument('--profile', choices=['quality'], default='quality')
    parser.add_argument('--input', type=Path, default=Path('input'))
    parser.add_argument('--output', type=Path, default=Path('output'))
    args = parser.parse_args()
    global PROFILE, EXTRA
    PROFILE = args.profile
    EXTRA = BASE_EXTRA|METHODS[args.method]
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output directories must differ')
    if not args.input.is_dir():
        parser.error(f'Input directory not found: {args.input}')
    files = sorted(path for path in args.input.iterdir()
                   if path.is_file() and path.suffix.lower() == '.wav')
    if not files:
        parser.error(f'No WAV files found in {args.input}')
    for index, source in enumerate(files, 1):
        process_file(source, args.output/source.name)
        if index % 100 == 0 or index == len(files):
            print(f'Processed {index}/{len(files)}')


if __name__ == '__main__':
    main()
