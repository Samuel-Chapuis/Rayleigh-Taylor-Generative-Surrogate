import torch


def continuous_sinusoidal_embedding(t, d, max_period=10_000.0):
    """Embedding sinusoidal pour un temps continu normalise dans ``[0, 1]``.

    Contrairement a :func:`sinusoidal_embedding`, cette fonction ne transforme
    pas ``t`` en indice entier. Elle peut donc etre utilisee par un VP-SDE.
    """
    if t.ndim != 1:
        t = t.reshape(-1)
    if d < 1:
        raise ValueError("La dimension de l'embedding doit etre positive.")

    half = d // 2
    if half == 0:
        return t[:, None]

    frequencies = torch.exp(
        -torch.log(torch.tensor(max_period, device=t.device, dtype=t.dtype))
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    angles = 2.0 * torch.pi * t[:, None] * frequencies[None, :]
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    if embedding.shape[1] < d:
        embedding = torch.cat((embedding, t[:, None]), dim=1)
    return embedding

def sinusoidal_embedding(n, d):
    """
    Construit l'embedding positionnel sinusoidal standard.

    Args:
        n: Nombre de positions ou pas de temps.
        d: Dimension de l'embedding.

    Returns:
        Tenseur de forme ``(n, d)`` alternant sinus et cosinus a plusieurs
        frequences.
    """
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding
