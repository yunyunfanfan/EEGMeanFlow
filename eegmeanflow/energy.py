"""Energy U-statistic for class-conditional relationship features."""
import math
from torch import Tensor

def smooth_distance(a: Tensor, b: Tensor, eps: float = 1e-4) -> Tensor:
    """Dimension-normalized Euclidean distance with finite zero gradient.

    Direct differences avoid cancellation in ||a||虏+||b||虏-2a.b near equality.
    eps is in standardized feature units, and the diagonal is exactly zero.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not a.shape[1]:
        raise ValueError('Expected two feature matrices with equal nonzero width')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('Positive finite distance epsilon required')
    d2 = (a[:, None, :] - b[None, :, :]).square().mean(-1)
    return (d2 + eps**2).sqrt() - eps


def energy_u(a: Tensor, b: Tensor, eps: float = 1e-4) -> Tensor:
    """Unbiased two-sample energy U-statistic, including fake-pair repulsion."""
    m, n = len(a), len(b)
    if min(m, n) < 2:
        raise ValueError('At least two rows per sample distribution required')
    aa, bb, ab = smooth_distance(a, a, eps), smooth_distance(b, b, eps), smooth_distance(a, b, eps)
    return (2 * ab.mean() - (aa.sum() - aa.diagonal().sum()) / (m*(m-1))
            - (bb.sum() - bb.diagonal().sum()) / (n*(n-1)))

