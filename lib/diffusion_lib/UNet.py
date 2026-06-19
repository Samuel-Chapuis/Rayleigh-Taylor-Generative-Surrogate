
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.diffusion_lib.embeding import sinusoidal_embedding

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

    def __init__(self, n_steps=1000, time_emb_dim=100, size=28, in_channels=1):
        """
        Initialise le U-Net.

        Args:
            n_steps (int, optional): Nombre total de pas de diffusion utilisés pour
                construire l'embedding temporel. Par défaut 1000.
            time_emb_dim (int, optional): Dimension de l'embedding temporel.
                Par défaut 100.
            size (int, optional): Taille des images carrées en entrée et sortie.
                Par défaut 28.
            in_channels (int, optional): Nombre de canaux en entrée et sortie.
                Par défaut 1.
        """
        super(UNet, self).__init__()
        if size < 12:
            raise ValueError("UNet nécessite size >= 12 avec cette architecture.")

        self.size = size
        self.in_channels = in_channels
        size_1 = size
        size_2 = self._conv_size(size_1, kernel_size=4, stride=2, padding=1)
        size_3 = self._conv_size(size_2, kernel_size=4, stride=2, padding=1)
        size_mid = self._conv_size(size_3, kernel_size=2, stride=1, padding=0)
        size_mid = self._conv_size(size_mid, kernel_size=4, stride=2, padding=1)
        if size_mid < 1:
            raise ValueError("UNet nécessite une taille intermédiaire positive.")

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        # Application de l'embedding temporel après chaque bloc pour conditionner le réseau par le temps à chaque niveau de la hiérarchie. On projette l'embedding temporel à la bonne dimension avant de l'ajouter à la sortie du bloc précédent.
        self.te1 = self._make_te(time_emb_dim, in_channels)
        self.b1 = nn.Sequential(
            Block((in_channels, size_1, size_1), in_channels, 10),
            Block((10, size_1, size_1), 10, 10),
            Block((10, size_1, size_1), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            Block((10, size_2, size_2), 10, 20),
            Block((20, size_2, size_2), 20, 20),
            Block((20, size_2, size_2), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            Block((20, size_3, size_3), 20, 40),
            Block((40, size_3, size_3), 40, 40),
            Block((40, size_3, size_3), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            Block((40, size_mid, size_mid), 40, 20),
            Block((20, size_mid, size_mid), 20, 20),
            Block((20, size_mid, size_mid), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            Block((80, size_3, size_3), 80, 40),
            Block((40, size_3, size_3), 40, 20),
            Block((20, size_3, size_3), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            Block((40, size_2, size_2), 40, 20),
            Block((20, size_2, size_2), 20, 10),
            Block((10, size_2, size_2), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            Block((20, size_1, size_1), 20, 10),
            Block((10, size_1, size_1), 10, 10),
            Block((10, size_1, size_1), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, in_channels, 3, 1, 1)

    def forward(self, x, t):
        """
        Prédit le bruit associé à une image bruitée.

        Args:
            x (torch.Tensor): Images d'entrée de forme ``(N, C, H, W)``.
            t (torch.Tensor): Pas de temps pour chaque image du lot.

        Returns:
            torch.Tensor: Carte de bruit prédite.
        """
        if x.shape[1:] != (self.in_channels, self.size, self.size):
            raise ValueError(
                f"UNet attend des tenseurs (N, {self.in_channels}, {self.size}, {self.size}), "
                f"mais a reçu {tuple(x.shape)}."
            )

        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, H, W)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, H/2, W/2)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, H/4, W/4)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))

        up1 = self._resize_to(self.up1(out_mid), out3)
        out4 = torch.cat((out3, up1), dim=1)  # (N, 80, H/4, W/4)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, H/4, W/4)

        up2 = self._resize_to(self.up2(out4), out2)
        out5 = torch.cat((out2, up2), dim=1)  # (N, 40, H/2, W/2)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, H/2, W/2)

        up3 = self._resize_to(self.up3(out5), out1)
        out = torch.cat((out1, up3), dim=1)  # (N, 20, H, W)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 10, H, W)

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

    def _conv_size(self, size, kernel_size, stride, padding, dilation=1):
        return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

    def _resize_to(self, x, target):
        if x.shape[-2:] == target.shape[-2:]:
            return x
        return F.interpolate(x, size=target.shape[-2:], mode="nearest")
