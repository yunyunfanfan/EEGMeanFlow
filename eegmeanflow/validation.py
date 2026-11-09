"""Train-fitted PCA32 and class-macro spectral/spatial MMD for model selection."""
import numpy as np
import torch
from sklearn.decomposition import PCA
from .features import descriptors
from .meanflow import sample_flow

def mmd(a, b):
    def distance(x, y):
        return np.maximum((x*x).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*x@y.T,0)
    aa,bb,ab = distance(a,a),distance(b,b),distance(a,b)
    positive = aa[np.triu_indices(len(a),1)]
    positive = positive[positive>1e-12]
    bandwidth = float(np.median(positive)) if len(positive) else 1.
    return float(np.mean([np.exp(-aa/(2*bandwidth*s*s)).mean()
        +np.exp(-bb/(2*bandwidth*s*s)).mean()-2*np.exp(-ab/(2*bandwidth*s*s)).mean()
        for s in [.5,1.,2.]]))

class Validator:
    def __init__(self, train, validation, per_class=128):
        x,y = validation
        features = descriptors(train.numpy()*100)[0]
        self.center = features.mean(0)
        self.scale = features.std(0).clip(1e-8)
        self.pca = PCA(n_components=32, svd_solver='full')
        projected = self.pca.fit_transform((features-self.center)/self.scale)
        self.projected_center = projected.mean(0)
        self.projected_scale = projected.std(0).clip(1e-8)
        rng = np.random.default_rng(41731)
        indices = np.concatenate([rng.choice(np.flatnonzero(y.numpy()==k),per_class,replace=False) for k in range(9)])
        self.reference = self.transform(descriptors(x[indices].numpy()*100)[0])
        self.labels = torch.arange(9).repeat_interleave(per_class)
        self.noise = torch.stack([torch.randn(32,2000,generator=torch.Generator().manual_seed(8888+i)) for i in range(len(self.labels))])
        self.per_class = per_class

    def transform(self, features):
        p = self.pca.transform((features-self.center)/self.scale)
        return (p-self.projected_center)/self.projected_scale

    @torch.no_grad()
    def score(self, model, device, batch_size=16):
        was_training = model.training
        model.eval()
        generated = []
        try:
            for z,y in zip(self.noise.split(batch_size),self.labels.split(batch_size)):
                generated.append(sample_flow(model,z.to(device),y.to(device)).cpu())
            x = torch.cat(generated).numpy()*100
            features = self.transform(descriptors(x)[0])
            return np.mean([mmd(self.reference[k*self.per_class:(k+1)*self.per_class],
                                features[k*self.per_class:(k+1)*self.per_class]) for k in range(9)]).item()
        finally:
            model.train(was_training)
