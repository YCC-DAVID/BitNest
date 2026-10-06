"""hadamard_transform compatibility layer: fast_hadamard_transform (CUDA extension) when available, otherwise a pure
torch butterfly (functionally equivalent, slower)."""
import torch

try:
    from fast_hadamard_transform import hadamard_transform as _fht
except Exception:
    _fht = None


def _torch_hadamard(x, scale=1.0):
    n = x.shape[-1]; assert n & (n - 1) == 0, "the torch fallback only supports power-of-two sizes"
    shp = x.shape; y = x.reshape(-1, n).float(); h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h); a = y[:, :, 0, :]; b = y[:, :, 1, :]; y = torch.stack([a + b, a - b], 2).view(-1, n); h *= 2
    return (y * scale).reshape(shp).to(x.dtype)


def hadamard_transform(x, scale=1.0):
    if _fht is not None and x.is_cuda:
        return _fht(x, scale=scale)
    return _torch_hadamard(x, scale)
