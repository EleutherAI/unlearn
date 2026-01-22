import torch
from torch import Tensor


@torch.inference_mode()
def stable_rank(A: Tensor) -> float:
    eps = torch.finfo(A.dtype).eps

    # Spectral norm
    _, S, _ = torch.svd_lowrank(A, q=1)
    spec = S[0]

    # Frobenius norm
    # frob = torch.linalg.matrix_norm(A, ord="fro")
    frob_sq = A.pow(2).sum()

    # Ratio of the squares
    return (frob_sq / (spec.pow(2) + eps)).item()


@torch.inference_mode()
def effective_rank(A: Tensor) -> float:
    eps = torch.finfo(A.dtype).eps

    # Get singular values
    S = torch.linalg.svdvals(A)

    # Normalize to get probability distribution
    S_norm = S / (S.sum() + eps)

    # Compute entropy (excluding zeros as lim
    # log(x) as x tends to 0 is 0)
    S_pos = S_norm[S_norm > eps]
    entropy = -(S_pos * S_pos.log()).sum()

    # Effective rank is exp(entropy)
    return entropy.exp().item()
