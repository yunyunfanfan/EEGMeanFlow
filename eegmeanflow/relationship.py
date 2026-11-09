"""Class-conditional cross-spectrum / local lag-covariance energy distance.

Only supplied real TRAIN rows fit the frozen feature scales and references.
No classifier, validation row, paired waveform target, or model-input re-reference.
"""
from __future__ import annotations
import torch
from torch import nn
from .energy import energy_u

GROUPS = ('cross_spectrum', 'local_lag_covariance')
BANDS = ((.3,4.),(4.,8.),(8.,13.),(13.,30.),(30.,45.))
LAGS = (0,2,10,20)  # at 200Hz: 0,10,50,100 ms


def raw_features(x, fs=200.):
    if x.ndim != 3 or x.shape[-1] < 400 or x.shape[1] < 2:
        raise ValueError('Expected B,C,T with C>=2,T>=400')
    if x.dtype not in (torch.float32,torch.float64) or not torch.isfinite(x).all():
        raise ValueError('Finite float32/64 EEG required')
    # Short-time Fourier transform, 2-second Hann windows with 50% overlap.
    # Segment demeaning is confined to this loss statistic.
    seg=x.unfold(-1,400,200);seg=seg-seg.mean(-1,keepdim=True)
    window=torch.hann_window(400,device=x.device,dtype=x.dtype)
    f=torch.fft.rfft(seg*window,dim=-1)
    hz=torch.fft.rfftfreq(400,1/fs,device=x.device)
    left,right=torch.triu_indices(x.shape[1],x.shape[1],1,device=x.device)
    parts=[]
    for low,high in BANDS:
        mask=(hz>=low)&(hz<high)
        a=f[:,left,...][...,mask];b=f[:,right,...][...,mask]
        cross=(a*b.conj()).mean((-1,-2))/(window.square().sum()*fs)
        # Signed real AND imaginary cross-spectrum retain relative phase.
        parts.extend((cross.real,cross.imag))
    spectrum=torch.cat(parts,dim=1)
    # One-second windows, half-second hop, explicit directed lag pairs.
    seg=x.unfold(-1,200,100);lag_parts=[]
    for lag in LAGS:
        a=seg if lag==0 else seg[...,:-lag]
        b=seg if lag==0 else seg[...,lag:]
        a=a-a.mean(-1,keepdim=True);b=b-b.mean(-1,keepdim=True)
        covariance=torch.einsum('bcwt,bdwt->bwcd',a,b)/a.shape[-1]
        lag_parts.append(covariance.flatten(1))
    return dict(cross_spectrum=spectrum,local_lag_covariance=torch.cat(lag_parts,dim=1))


class EEGCrossLag(nn.Module):
    def __init__(self,banks,labels,scales,means,stds,metadata):
        super().__init__();self.metadata=metadata
        self.register_buffer('labels',labels.long().clone())
        for name in GROUPS:
            for prefix,source in [('bank',banks),('scale',scales),('mean',means),('std',stds)]:
                self.register_buffer(prefix+'_'+name,source[name].detach().clone())
        self.classes=tuple(torch.unique(labels).tolist())
        self.indices={k:torch.where(labels.cpu()==k)[0] for k in self.classes}

    @classmethod
    def from_training(cls,x,y,batch_size=16):
        if x.device.type!='cpu' or y.device.type!='cpu' or len(x)!=len(y):
            raise ValueError('Supplied CPU training rows and labels required')
        if torch.unique(y,return_counts=True)[1].min()<2:raise ValueError('Two rows per class required')
        parts={g:[] for g in GROUPS};banks={};scales={};means={};stds={}
        with torch.no_grad():
            for batch in x.split(batch_size):
                for k,v in raw_features(batch).items():parts[k].append(v)
            for k in GROUPS:
                raw=torch.cat(parts.pop(k));pooled=raw.abs().median().clamp_min(1e-10)
                scales[k]=raw.abs().median(0).values.clamp_min(.05*pooled).clamp_min(1e-10)
                v=torch.asinh(raw/scales[k]);means[k]=v.mean(0)
                stds[k]=v.std(0,unbiased=False).clamp_min(.05)
                banks[k]=(v-means[k])/stds[k]
        meta=dict(version='eeg_cross_lag_v1',fit_split='train_only',input_units='uV/100',
            input_shape=list(x.shape[1:]),sample_rate=200,bands_hz=[list(v) for v in BANDS],lags_samples=list(LAGS),
            cross_spectrum='off-diagonal signed real+imaginary; mean Hann400/hop200 cross periodogram',
            lag_covariance='all directed channel pairs, demeaning per truncated lag segment; window200/hop100',
            normalization='asinh(raw/train median abs with group-relative .05 floor); train population zscore std floor .05',
            estimator='equal class/group energy U-statistic, self-pairs excluded, negative estimates retained',
            feature_dimensions={k:banks[k].shape[1] for k in GROUPS})
        return cls(banks,y,scales,means,stds,meta)

    def loss(self,fake,labels,generator,refs_per_class=64):
        if tuple(fake.shape[1:])!=tuple(self.metadata['input_shape']):raise ValueError('Shape mismatch')
        classes,counts=torch.unique(labels,return_counts=True)
        if tuple(classes.tolist())!=self.classes or counts.min()<2:raise ValueError('Every class needs >=2 fake rows')
        raw=raw_features(fake);terms={g:[] for g in GROUPS}
        for g in GROUPS:
            raw[g]=(torch.asinh(raw[g]/getattr(self,'scale_'+g))-getattr(self,'mean_'+g))/getattr(self,'std_'+g)
        for k in self.classes:
            pool=self.indices[k]
            index=pool[torch.randperm(len(pool),generator=generator)[:min(refs_per_class,len(pool))]].to(fake.device)
            for g in GROUPS:
                terms[g].append(energy_u(raw[g][labels==k],getattr(self,'bank_'+g)[index]))
        groups={g:torch.stack(v).mean() for g,v in terms.items()}
        return dict(loss=torch.stack(list(groups.values())).mean(),groups=groups)

    def get_extra_state(self):return self.metadata
    def set_extra_state(self,state):self.metadata=state
