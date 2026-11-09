"""MeanFlow loss and one-step sampling. Based on https://arxiv.org/abs/2505.13447."""
from contextlib import contextmanager
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class MeanFlowConfig:
    time_mean: float = -2.0
    time_std: float = 2.0
    off_diagonal_ratio: float = 0.75
    norm_power: float = 0.75
    norm_eps: float = 1e-3


def sample_intervals(batch, device, config=MeanFlowConfig()):
    if not 0 <= config.off_diagonal_ratio <= 1 or config.time_std <= 0:
        raise ValueError('Invalid MeanFlow interval sampling parameters')
    a = torch.sigmoid(torch.randn(batch, device=device) * config.time_std + config.time_mean)
    b = torch.sigmoid(torch.randn(batch, device=device) * config.time_std + config.time_mean)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    diagonal = torch.rand(batch, device=device) >= config.off_diagonal_ratio
    return t, torch.where(diagonal, t, r)


def drop_labels_once(model, labels):
    embedder = getattr(model, 'y_embedder', None)
    if model.training and embedder is not None and embedder.dropout_prob:
        mask = torch.rand(labels.shape, device=labels.device) < embedder.dropout_prob
        return torch.where(mask, embedder.num_classes, labels)
    return labels


@contextmanager
def jvp_attention(model):
    # PyTorch fused SDPA forward AD is not universally supported. Both the
    # prediction and its derivative use identical mathematical attention here.
    # Sampling retains the original fused backend and needs no JVP.
    modules = [(m, m.fused_attn) for m in model.modules() if hasattr(m, 'fused_attn')]
    try:
        for module, _ in modules:
            module.fused_attn = False
        yield
    finally:
        for module, value in modules:
            module.fused_attn = value


def meanflow_loss(model, x, labels, *, config=MeanFlowConfig(), noise=None, t=None, r=None):
    if config.norm_power < 0 or config.norm_eps <= 0:
        raise ValueError('Invalid MeanFlow adaptive weighting')
    if (t is None) != (r is None):
        raise ValueError('Provide both t and r or neither')
    noise = torch.randn_like(x) if noise is None else noise
    if t is None:
        t, r = sample_intervals(len(x), x.device, config)
    h = (t-r).reshape(-1, 1, 1)
    z = (1-t.reshape(-1, 1, 1))*x + t.reshape(-1, 1, 1)*noise
    velocity = noise-x
    fixed_labels = drop_labels_once(model, labels)

    def field(z_value, t_value, r_value):
        return model(z_value, t_value, fixed_labels, r=r_value)

    with jvp_attention(model):
        prediction = field(z, t, r)
        # Separate no-grad JVP is an officially documented alternative to
        # compiled JVP; no second-order parameter graph is retained.
        with torch.no_grad():
            _, derivative = torch.func.jvp(field, (z, t, r),
                (velocity, torch.ones_like(t), torch.zeros_like(r)))
        target = (velocity-h*derivative).detach()
        per_sample_sum = (prediction-target).square().flatten(1).sum(1)
        weight = (per_sample_sum.detach()+config.norm_eps).pow(config.norm_power)
        loss = (per_sample_sum/weight).mean()
    return dict(loss=loss, velocity_mse=(prediction.detach()-velocity).square().mean(),
                target_mse=(prediction.detach()-target).square().mean(),
                diagonal_fraction=(r==t).float().mean(),
                target=target, derivative=derivative.detach())


@torch.no_grad()
def sample_flow(model, noise, labels, *, steps=1):
    if model.training:
        raise ValueError('Set model.eval() before sampling')
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('Steps must be a positive integer')
    z = noise.clone()
    for index in range(steps):
        t = torch.full((len(z),), 1-index/steps, device=z.device, dtype=z.dtype)
        r = torch.full_like(t, 1-(index+1)/steps)
        velocity = model(z, t, labels, r=r) if model.meanflow else model(z, t, labels)
        z = z - velocity/steps
    return z
