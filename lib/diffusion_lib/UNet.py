
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

    def __init__(
        self,
        n_steps=1000,
        time_emb_dim=100,
        size=28,
        in_channels=1,
        depth=3,
        blocks_per_level=3,
        base_channels=10,
        channel_multipliers=None,
    ):
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
            depth (int, optional): Nombre de niveaux de descente/remontée du U-Net,
                donc aussi nombre de pooling par convolution stride 2. Par défaut 3.
            blocks_per_level (int, optional): Nombre de blocs convolutionnels à
                chaque niveau. Un ``Block`` contient deux convolutions. Par défaut 3.
            base_channels (int, optional): Nombre de canaux du premier niveau.
                Par défaut 10.
            channel_multipliers (list[int] | tuple[int, ...] | None, optional):
                Multiplicateurs de canaux par niveau. Si ``None``, utilise
                ``[1, 2, 4, ...]``. Par défaut None.
        """
        super(UNet, self).__init__()
        if depth < 1:
            raise ValueError("UNet nécessite depth >= 1.")
        if blocks_per_level < 1:
            raise ValueError("UNet nécessite blocks_per_level >= 1.")
        if base_channels < 1:
            raise ValueError("UNet nécessite base_channels >= 1.")

        self.size = size
        self.in_channels = in_channels
        self.depth = depth
        self.blocks_per_level = blocks_per_level
        self.base_channels = base_channels

        if channel_multipliers is None:
            channel_multipliers = tuple(2 ** level for level in range(depth))
        if len(channel_multipliers) != depth:
            raise ValueError(
                "channel_multipliers doit contenir exactement depth valeurs "
                f"({depth} attendues, {len(channel_multipliers)} reçues)."
            )
        self.channels = tuple(base_channels * multiplier for multiplier in channel_multipliers)

        self.legacy_default = (
            depth == 3
            and blocks_per_level == 3
            and base_channels == 10
            and tuple(channel_multipliers) == (1, 2, 4)
        )
        if self.legacy_default:
            self._init_legacy_default(n_steps, time_emb_dim, size, in_channels)
            return

        spatial_sizes = [size]
        for _ in range(depth):
            next_size = self._conv_size(spatial_sizes[-1], kernel_size=4, stride=2, padding=1)
            if next_size < 1:
                raise ValueError(
                    "UNet nécessite une taille intermédiaire positive. "
                    f"Réduis depth pour size={size}."
                )
            spatial_sizes.append(next_size)
        self.spatial_sizes = tuple(spatial_sizes)

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # Application de l'embedding temporel à chaque niveau de la hiérarchie.
        self.encoder_time = nn.ModuleList()
        self.encoder_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        current_channels = in_channels
        for level, out_channels in enumerate(self.channels):
            level_size = spatial_sizes[level]
            self.encoder_time.append(self._make_te(time_emb_dim, current_channels))
            self.encoder_blocks.append(
                self._make_block_stack(
                    level_size,
                    current_channels,
                    out_channels,
                    blocks_per_level,
                )
            )
            self.downs.append(nn.Conv2d(out_channels, out_channels, 4, 2, 1))
            current_channels = out_channels

        self.mid_time = self._make_te(time_emb_dim, current_channels)
        self.mid_block = self._make_block_stack(
            spatial_sizes[-1],
            current_channels,
            current_channels,
            blocks_per_level,
        )

        self.ups = nn.ModuleList()
        self.decoder_time = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for level in reversed(range(depth)):
            skip_channels = self.channels[level]
            decoder_in_channels = 2 * skip_channels
            decoder_out_channels = self.channels[level - 1] if level > 0 else self.channels[0]

            self.ups.append(nn.ConvTranspose2d(current_channels, skip_channels, 4, 2, 1))
            self.decoder_time.append(self._make_te(time_emb_dim, decoder_in_channels))
            self.decoder_blocks.append(
                self._make_block_stack(
                    spatial_sizes[level],
                    decoder_in_channels,
                    decoder_out_channels,
                    blocks_per_level,
                    final_normalize=level != 0,
                )
            )
            current_channels = decoder_out_channels

        self.conv_out = nn.Conv2d(current_channels, in_channels, 3, 1, 1)

    def _init_legacy_default(self, n_steps, time_emb_dim, size, in_channels):
        if size < 12:
            raise ValueError("UNet nécessite size >= 12 avec cette architecture.")

        size_1 = size
        size_2 = self._conv_size(size_1, kernel_size=4, stride=2, padding=1)
        size_3 = self._conv_size(size_2, kernel_size=4, stride=2, padding=1)
        size_mid = self._conv_size(size_3, kernel_size=2, stride=1, padding=0)
        size_mid = self._conv_size(size_mid, kernel_size=4, stride=2, padding=1)
        if size_mid < 1:
            raise ValueError("UNet nécessite une taille intermédiaire positive.")

        self.spatial_sizes = (size_1, size_2, size_3, size_mid)

        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

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

        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            Block((40, size_mid, size_mid), 40, 20),
            Block((20, size_mid, size_mid), 20, 20),
            Block((20, size_mid, size_mid), 20, 40)
        )

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
        if self.legacy_default:
            return self._forward_legacy_default(x, t, n)

        skips = []
        out = x
        for time_proj, block, down in zip(self.encoder_time, self.encoder_blocks, self.downs):
            out = block(out + time_proj(t).reshape(n, -1, 1, 1))
            skips.append(out)
            out = down(out)

        out = self.mid_block(out + self.mid_time(t).reshape(n, -1, 1, 1))

        for up, time_proj, block, skip in zip(
            self.ups,
            self.decoder_time,
            self.decoder_blocks,
            reversed(skips),
        ):
            out = self._resize_to(up(out), skip)
            out = torch.cat((skip, out), dim=1)
            out = block(out + time_proj(t).reshape(n, -1, 1, 1))

        out = self.conv_out(out)

        return out

    def _forward_legacy_default(self, x, t, n):
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))

        up1 = self._resize_to(self.up1(out_mid), out3)
        out4 = torch.cat((out3, up1), dim=1)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))

        up2 = self._resize_to(self.up2(out4), out2)
        out5 = torch.cat((out2, up2), dim=1)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))

        up3 = self._resize_to(self.up3(out5), out1)
        out = torch.cat((out1, up3), dim=1)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))

        return self.conv_out(out)

    def _make_block_stack(self, size, in_channels, out_channels, num_blocks, final_normalize=True):
        blocks = []
        current_channels = in_channels
        for block_idx in range(num_blocks):
            normalize = final_normalize or block_idx < num_blocks - 1
            blocks.append(
                Block(
                    (current_channels, size, size),
                    current_channels,
                    out_channels,
                    normalize=normalize,
                )
            )
            current_channels = out_channels
        return nn.Sequential(*blocks)

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
