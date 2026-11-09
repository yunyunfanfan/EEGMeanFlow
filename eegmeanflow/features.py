"""Spectral/spatial features for validation model selection."""
import numpy as np
from scipy import signal

BANDS=((.3,4),(4,8),(8,13),(13,30),(30,45))


def descriptors(waveforms,fs=200,batch_size=32):
    if waveforms.ndim!=3 or not np.isfinite(waveforms).all():
        raise ValueError('Expected finite N x C x T waveforms')
    if waveforms.shape[-1]<400:raise ValueError('At least 2 seconds at 200Hz required')
    features=[];powers=[];common_mode=[]
    upper=np.triu_indices(waveforms.shape[1],1)
    for start in range(0,len(waveforms),batch_size):
        x=waveforms[start:start+batch_size].astype(np.float64)
        frequencies,psd=signal.welch(x,fs=fs,nperseg=400,noverlap=200,axis=-1)
        df=float(frequencies[1]-frequencies[0])
        power=[]
        for low,high in BANDS:
            mask=(frequencies>=low)&(frequencies<high if high<45 else frequencies<=high)
            # Welch returns density per Hz. Sum complete disjoint FFT bins;
            # separate trapezoids lose half of both boundary bins per band.
            power.append(psd[...,mask].sum(axis=-1)*df)
        power=np.stack(power,-1)
        centered=x-x.mean(-1,keepdims=True)
        std=np.sqrt(np.mean(centered**2,-1)).clip(1e-12)
        normalized=centered/std[...,None]
        correlation=normalized@normalized.transpose(0,2,1)/x.shape[-1]
        spatial=correlation[:,upper[0],upper[1]]
        features.append(np.concatenate([np.log10(power.clip(1e-12)).reshape(len(x),-1),spatial],1))
        powers.append(power)
        common_mode.extend((np.sqrt(np.mean(centered.mean(1)**2,axis=-1))/
                            np.sqrt(np.mean(centered**2,axis=(1,2))).clip(1e-12)).tolist())
    return np.concatenate(features),np.concatenate(powers),np.asarray(common_mode)

