
import torch
import torch.nn as nn

from diffusion_lib.embeding import sinusoidal_embedding

class Block(nn.Module):
    """
    Bloc convolutionnel utilisé dans le U-Net.

    Le bloc applique éventuellement une normalisation par couche, puis deux
    convolutions successives séparées par une activation.
    """

    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        """
        Initialise un bloc convolutionnel.

        Args:
            shape (tuple[int, int, int]): Forme attendue pour la normalisation
                par couche sous la forme ``(C, H, W)``.
            in_c (int): Nombre de canaux en entrée.
            out_c (int): Nombre de canaux en sortie.
            kernel_size (int, optional): Taille du noyau de convolution. Par défaut 3.
            stride (int, optional): Pas de la convolution. Par défaut 1.
            padding (int, optional): Remplissage appliqué aux convolutions.
                Par défaut 1.
            activation (torch.nn.Module | None, optional): Fonction d'activation
                à utiliser. Si `None`, `SiLU` est employée.
            normalize (bool, optional): Indique si la normalisation par couche est
                activée. Par défaut True.
        """
        super(Block, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        """
        Propulse les données à travers le bloc convolutionnel.

        Args:
            x (torch.Tensor): Tenseur d'entrée.

        Returns:
            torch.Tensor: Tenseur transformé par le bloc.
        """
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out

class UNet(nn.Module):
    """
    Architecture U-Net conditionnée par le temps.

    Le réseau reçoit une image bruitée et un pas de diffusion, puis prédit le
    bruit associé à cette image.
    """

    def __init__(self, n_steps=1000, time_emb_dim=100):
        """
        Initialise le U-Net.

        Args:
            n_steps (int, optional): Nombre total de pas de diffusion utilisés pour
                construire l'embedding temporel. Par défaut 1000.
            time_emb_dim (int, optional): Dimension de l'embedding temporel.
                Par défaut 100.
        """
        super(UNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        # Application de l'embedding temporel après chaque bloc pour conditionner le réseau par le temps à chaque niveau de la hiérarchie. On projette l'embedding temporel à la bonne dimension avant de l'ajouter à la sortie du bloc précédent.
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            Block((1, 28, 28), 1, 10),
            Block((10, 28, 28), 10, 10),
            Block((10, 28, 28), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            Block((10, 14, 14), 10, 20),
            Block((20, 14, 14), 20, 20),
            Block((20, 14, 14), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            Block((20, 7, 7), 20, 40),
            Block((40, 7, 7), 40, 40),
            Block((40, 7, 7), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            Block((40, 3, 3), 40, 20),
            Block((20, 3, 3), 20, 20),
            Block((20, 3, 3), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            Block((80, 7, 7), 80, 40),
            Block((40, 7, 7), 40, 20),
            Block((20, 7, 7), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            Block((40, 14, 14), 40, 20),
            Block((20, 14, 14), 20, 10),
            Block((10, 14, 14), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            Block((20, 28, 28), 20, 10),
            Block((10, 28, 28), 10, 10),
            Block((10, 28, 28), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        """
        Prédit le bruit associé à une image bruitée.

        Args:
            x (torch.Tensor): Images d'entrée de forme ``(N, C, H, W)``.
            t (torch.Tensor): Pas de temps pour chaque image du lot.

        Returns:
            torch.Tensor: Carte de bruit prédite.
        """
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        """
        Construit un petit réseau pour projeter l'embedding temporel.

        Args:
            dim_in (int): Dimension d'entrée.
            dim_out (int): Dimension de sortie.

        Returns:
            torch.nn.Sequential: Bloc de projection de l'embedding temporel.
        """
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
    