"""Smoke-check default raw3 CLI format and included real-data waveforms.

Synthetic signals below test the I/O contract only, not perceptual quality.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import soundfile as sf


def main():
    folder=Path(__file__).resolve().parent
    checks=[]
    with TemporaryDirectory() as temporary:
        root=Path(temporary);source=root/'input';output=root/'output';source.mkdir()
        rng=np.random.default_rng(20261117)
        for name,sr,n,channels in [('mono',16000,48000,1),('stereo',16000,48000,2),('44100',44100,44100,1),('short',16000,800,1),('silent',16000,48000,1)]:
            t=np.arange(n)/sr
            x=np.zeros((n,channels)) if name=='silent' else np.column_stack([.1*np.sin(2*np.pi*(180+30*c)*t)+.01*rng.standard_normal(n) for c in range(channels)])
            sf.write(source/(name+'.WAV'),x,sr,subtype='PCM_16')
        examples=folder/'listening_examples/raw3'
        for path in sorted(examples.glob('*_noisy.wav')):
            x,sr=sf.read(path);sf.write(source/path.name,x,sr,subtype=sf.info(path).subtype)
        subprocess.run([sys.executable,str(folder/'denoise.py'),'--input',str(source),'--output',str(output)],check=True)
        inputs=sorted(source.iterdir());outputs=sorted(output.iterdir())
        assert {p.name for p in inputs}=={p.name for p in outputs}
        for path in inputs:
            original=sf.info(path);processed=sf.info(output/path.name);y,_=sf.read(output/path.name)
            assert (original.samplerate,original.channels,original.frames,original.subtype)==(processed.samplerate,processed.channels,processed.frames,processed.subtype)
            assert np.isfinite(y).all()
            row=dict(file=path.name,sample_rate=processed.samplerate,channels=processed.channels,frames=processed.frames,passed=True)
            if path.name.endswith('_noisy.wav'):
                expected,_=sf.read(examples/path.name.replace('_noisy.wav','_raw3.wav'))
                assert np.array_equal(y,expected),path.name
                row['matches_evaluated_raw3']=True
            checks.append(row)
    destination=folder/'results/io_checks.json';destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('Default raw3 CLI checks passed:',len(checks))


if __name__=='__main__':main()
