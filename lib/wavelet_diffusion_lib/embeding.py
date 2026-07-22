import torch

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
