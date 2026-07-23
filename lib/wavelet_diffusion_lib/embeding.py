import torch

def sinusoidal_embedding(t, d):
    """
    Construit l'embedding positionnel sinusoidal standard.

    Args:
        t: Temps de forme ``(N,)``. Les temps peuvent être continus.
        d: Dimension de l'embedding.

    Returns:
        Tenseur de forme ``(n, d)`` alternant sinus et cosinus a plusieurs
        frequences.
    """
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32)
    t = t.reshape(-1).float()
    half = d // 2
    if half == 0:
        return t[:, None]
    frequencies = torch.exp(
        -torch.log(torch.tensor(10_000.0, device=t.device))
        * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t[:, None] * frequencies[None]
    embedding = torch.cat((torch.sin(args), torch.cos(args)), dim=1)
    if d % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=1)
    return embedding
