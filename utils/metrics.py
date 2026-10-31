import torch
import numpy as np

def compute_cus(uncertainties):
    return float(torch.tensor(uncertainties).mean().item())

def compute_confidence_entropy(probabilities):
    """Computes entropy of a probability distribution (confidence entropy)."""
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    return -np.sum(probabilities * np.log(probabilities))

def compute_zti(cus_scores, epsilon=1e-6):
    cus_tensor = torch.tensor(cus_scores, dtype=torch.float32)
    if len(cus_tensor) < 2:
        return 1.0

    mean_cus = cus_tensor.mean()
    std_cus = cus_tensor.std(unbiased=False)
    max_cus = torch.maximum(cus_tensor.max(), torch.tensor(epsilon))

    zti = 1 - (torch.log1p(torch.abs(mean_cus - std_cus)) / torch.log1p(max_cus))
    return torch.clamp(zti, 0.0, 1.0).item()

