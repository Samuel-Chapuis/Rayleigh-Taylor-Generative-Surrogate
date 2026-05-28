import torch


def get_best_device():
    """
    Détermine le meilleur dispositif de calcul disponible (GPU, MPS ou CPU).

    La priorité est donnée à CUDA, puis à MPS, puis au CPU.

    Returns:
        torch.device: le dispositif de calcul à utiliser
    """    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")