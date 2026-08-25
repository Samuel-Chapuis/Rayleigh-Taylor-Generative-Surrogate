
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.diffusion_lib.embeding import continuous_sinusoidal_embedding, sinusoidal_embedding


class HorizontalCircularConv2d(nn.Module):
    """Convolution with circular padding along x and zero padding along y.

    The Rayleigh--Taylor domain is periodic horizontally but not vertically.
    PyTorch's native ``padding_mode='circular'`` wraps both axes, which would
    incorrectly couple the top and bottom boundaries.  This layer instead
    applies the two boundary conditions explicitly before an unpadded
    convolution.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
        bias=True,
    ):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding)
        if len(padding) != 2:
            raise ValueError("padding doit etre un entier ou un couple (vertical, horizontal).")
        self.padding_y, self.padding_x = padding
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias,
        )

    def forward(self, x):
        if self.padding_x:
            x = F.pad(x, (self.padding_x, self.padding_x, 0, 0), mode="circular")
        if self.padding_y:
            x = F.pad(x, (0, 0, self.padding_y, self.padding_y), mode="constant", value=0.0)
        return self.conv(x)


def make_conv2d(
    in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    padding_mode="zeros",
):
    """Build a convolution respecting the selected physical boundary condition."""
    if padding_mode == "zeros":
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    if padding_mode == "horizontal_circular":
        return HorizontalCircularConv2d(
            in_channels, out_channels, kernel_size, stride, padding,
        )
    raise ValueError("padding_mode doit etre 'zeros' ou 'horizontal_circular'.")


class BlurPool2d(nn.Module):
    """Fixed separable [1, 2, 1]^2 low-pass filter followed by decimation.

    The filter suppresses frequencies above the post-decimation Nyquist limit.
    Horizontal padding is circular and vertical padding remains zero, matching
    ``HorizontalCircularConv2d`` and the RT boundary conditions.
    """

    def __init__(self, channels, padding_mode="zeros"):
        super().__init__()
        if padding_mode not in {"zeros", "horizontal_circular"}:
            raise ValueError("padding_mode doit etre 'zeros' ou 'horizontal_circular'.")
        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel_2d = torch.outer(kernel_1d, kernel_1d).div_(16.0)
        self.register_buffer("kernel", kernel_2d.reshape(1, 1, 3, 3))
        self.channels = channels
        self.padding_mode = padding_mode

    def forward(self, x):
        if self.padding_mode == "horizontal_circular":
            x = F.pad(x, (1, 1, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, 1, 1), mode="constant", value=0.0)
        else:
            x = F.pad(x, (1, 1, 1, 1), mode="constant", value=0.0)
        kernel = self.kernel.to(dtype=x.dtype).expand(self.channels, 1, 3, 3)
        return F.conv2d(x, kernel, stride=2, groups=self.channels)


class BlurPoolDownsample(nn.Module):
    """Learn features at full resolution, low-pass them, then subsample by 2."""

    def __init__(self, channels, padding_mode="zeros"):
        super().__init__()
        self.projection = make_conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1,
            padding_mode=padding_mode,
        )
        self.blur = BlurPool2d(channels, padding_mode=padding_mode)

    def forward(self, x):
        return self.blur(self.projection(x))


class Block(nn.Module):
    """
    Bloc convolutionnel utilisé dans le U-Net.

    Le bloc applique éventuellement une normalisation par couche, puis deux
    convolutions successives séparées par une activation.
    """

    def __init__(
        self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1,
        activation=None, normalize=True, norm_type="layer", padding_mode="zeros"
    ):
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
        if norm_type == "layer":
            # Legacy path: affine parameters depend on every spatial position.
            self.ln = nn.LayerNorm(shape)
        elif norm_type == "group":
            # No spatially indexed affine parameters.  The group count also
            # handles the 4-channel conditional wavelet input exactly.
            n_groups = min(8, in_c)
            while in_c % n_groups != 0:
                n_groups -= 1
            self.norm = nn.GroupNorm(n_groups, in_c)
        else:
            raise ValueError("norm_type doit etre 'layer' ou 'group'.")
        self.conv1 = make_conv2d(
            in_c, out_c, kernel_size, stride, padding, padding_mode=padding_mode,
        )
        self.conv2 = make_conv2d(
            out_c, out_c, kernel_size, stride, padding, padding_mode=padding_mode,
        )
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize
        self.norm_type = norm_type

    def forward(self, x):
        """
        Propulse les données à travers le bloc convolutionnel.

        Args:
            x (torch.Tensor): Tenseur d'entrée.

        Returns:
            torch.Tensor: Tenseur transformé par le bloc.
        """
        if self.normalize:
            out = self.ln(x) if self.norm_type == "layer" else self.norm(x)
        else:
            out = x
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
        out_channels=None,
        depth=3,
        blocks_per_level=3,
        base_channels=10,
        channel_multipliers=None,
        continuous_time=False,
        norm_type="layer",
        upsample_mode="conv_transpose",
        padding_mode="zeros",
        downsample_mode="stride_conv",
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
            in_channels (int, optional): Nombre de canaux en entrée.
                Par défaut 1.
            out_channels (int | None, optional): Nombre de canaux prédits en
                sortie. Si ``None``, reprend ``in_channels``. Par défaut None.
            depth (int, optional): Nombre de niveaux de descente/remontée du U-Net,
                donc aussi nombre de pooling par convolution stride 2. Par défaut 3.
            blocks_per_level (int, optional): Nombre de blocs convolutionnels à
                chaque niveau. Un ``Block`` contient deux convolutions. Par défaut 3.
            base_channels (int, optional): Nombre de canaux du premier niveau.
                Par défaut 10.
            channel_multipliers (list[int] | tuple[int, ...] | None, optional):
                Multiplicateurs de canaux par niveau. Si ``None``, utilise
                ``[1, 2, 4, ...]``. Par défaut None.
            continuous_time (bool, optional): Utilise un embedding differentiable
                pour ``t`` dans ``[0, 1]``. A conserver a ``False`` pour DDPM.
        """
        super(UNet, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if depth < 1:
            raise ValueError("UNet nécessite depth >= 1.")
        if blocks_per_level < 1:
            raise ValueError("UNet nécessite blocks_per_level >= 1.")
        if base_channels < 1:
            raise ValueError("UNet nécessite base_channels >= 1.")
        if out_channels < 1:
            raise ValueError("UNet nécessite out_channels >= 1.")
        if norm_type not in {"layer", "group"}:
            raise ValueError("norm_type doit etre 'layer' ou 'group'.")
        if upsample_mode not in {"conv_transpose", "interpolate"}:
            raise ValueError(
                "upsample_mode doit etre 'conv_transpose' ou 'interpolate'."
            )
        if padding_mode not in {"zeros", "horizontal_circular"}:
            raise ValueError("padding_mode doit etre 'zeros' ou 'horizontal_circular'.")
        if downsample_mode not in {"stride_conv", "blurpool"}:
            raise ValueError("downsample_mode doit etre 'stride_conv' ou 'blurpool'.")

        self.size = size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.blocks_per_level = blocks_per_level
        self.base_channels = base_channels
        self.continuous_time = continuous_time
        self.time_embed_dim = time_emb_dim
        self.norm_type = norm_type
        self.upsample_mode = upsample_mode
        self.padding_mode = padding_mode
        self.downsample_mode = downsample_mode

        if channel_multipliers is None:
            channel_multipliers = tuple(2 ** level for level in range(depth))
        if len(channel_multipliers) != depth:
            raise ValueError(
                "channel_multipliers doit contenir exactement depth valeurs "
                f"({depth} attendues, {len(channel_multipliers)} reçues)."
            )
        self.channels = tuple(base_channels * multiplier for multiplier in channel_multipliers)

        # spatial_sizes[level] est la resolution avant le downsampling du niveau.
        # La derniere valeur est la resolution du bottleneck apres depth descentes.
        spatial_sizes = [size]
        for _ in range(depth):
            down_kernel = 4 if downsample_mode == "stride_conv" else 3
            next_size = self._conv_size(
                spatial_sizes[-1], kernel_size=down_kernel, stride=2, padding=1,
            )
            if next_size < 1:
                raise ValueError(
                    "UNet nécessite une taille intermédiaire positive. "
                    f"Réduis depth pour size={size}."
                )
            spatial_sizes.append(next_size)
        self.spatial_sizes = tuple(spatial_sizes)

        # Embedding temporel fixe: t -> vecteur sinusoidal, ensuite projete au
        # nombre de canaux attendu par chaque niveau.
        if continuous_time:
            self.time_embed = None
        else:
            self.time_embed = nn.Embedding(n_steps, time_emb_dim)
            self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
            self.time_embed.requires_grad_(False)

        # Encodeur: un niveau = projection temporelle + blocks_per_level blocs +
        # une convolution stride 2. Les sorties avant downsampling sont gardees
        # pour les skip connections.
        self.encoder_time = nn.ModuleList()
        self.encoder_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        current_channels = in_channels
        for level, level_channels in enumerate(self.channels):
            level_size = spatial_sizes[level]
            self.encoder_time.append(self._make_te(time_emb_dim, current_channels))
            self.encoder_blocks.append(
                self._make_block_stack(
                    level_size,
                    current_channels,
                    level_channels,
                    blocks_per_level,
                    norm_type=norm_type,
                    padding_mode=padding_mode,
                )
            )
            if downsample_mode == "stride_conv":
                self.downs.append(
                    make_conv2d(
                        level_channels, level_channels, 4, 2, 1,
                        padding_mode=padding_mode,
                    )
                )
            else:
                self.downs.append(BlurPoolDownsample(level_channels, padding_mode=padding_mode))
            current_channels = level_channels

        # Bottleneck a la resolution la plus basse. Il a la meme largeur que le
        # dernier niveau encodeur.
        self.mid_time = self._make_te(time_emb_dim, current_channels)
        self.mid_block = self._make_block_stack(
            spatial_sizes[-1],
            current_channels,
            current_channels,
            blocks_per_level,
            norm_type=norm_type,
            padding_mode=padding_mode,
        )

        # Decodeur: chaque niveau remonte d'un facteur 2, concatene le skip de
        # meme resolution, puis applique blocks_per_level blocs.
        self.ups = nn.ModuleList()
        self.decoder_time = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for level in reversed(range(depth)):
            skip_channels = self.channels[level]
            decoder_in_channels = 2 * skip_channels
            decoder_out_channels = self.channels[level - 1] if level > 0 else self.channels[0]

            if upsample_mode == "conv_transpose":
                up = nn.ConvTranspose2d(current_channels, skip_channels, 4, 2, 1)
            else:
                up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    make_conv2d(
                        current_channels, skip_channels, 3, 1, 1,
                        padding_mode=padding_mode,
                    ),
                )
            self.ups.append(up)
            self.decoder_time.append(self._make_te(time_emb_dim, decoder_in_channels))
            self.decoder_blocks.append(
                self._make_block_stack(
                    spatial_sizes[level],
                    decoder_in_channels,
                    decoder_out_channels,
                    blocks_per_level,
                    final_normalize=level != 0,
                    norm_type=norm_type,
                    padding_mode=padding_mode,
                )
            )
            current_channels = decoder_out_channels

        self.conv_out = make_conv2d(
            current_channels, self.out_channels, 3, 1, 1, padding_mode=padding_mode,
        )

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

        if self.continuous_time:
            t = continuous_sinusoidal_embedding(t.float(), self.time_embed_dim)
        else:
            t = self.time_embed(t.long().reshape(-1))
        n = len(x)

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

    def _make_block_stack(
        self, size, in_channels, out_channels, num_blocks,
        final_normalize=True, norm_type="layer", padding_mode="zeros"
    ):
        """
        Construit une pile de blocs convolutionnels a resolution fixe.

        Args:
            size: Taille spatiale carree traitee par les blocs.
            in_channels: Nombre de canaux en entree du premier bloc.
            out_channels: Nombre de canaux en sortie de chaque bloc.
            num_blocks: Nombre de blocs a empiler.
            final_normalize: Desactive la normalisation du dernier bloc si faux.

        Returns:
            Sequence PyTorch de blocs convolutionnels.
        """
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
                    norm_type=norm_type,
                    padding_mode=padding_mode,
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
        """
        Calcule la taille spatiale de sortie d'une convolution 2D.

        La formule suit la convention PyTorch pour une dimension spatiale.
        """
        return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

    def _resize_to(self, x, target):
        """
        Redimensionne ``x`` vers la resolution spatiale de ``target`` si besoin.

        Ce correctif evite les erreurs de concaténation des skip connections
        quand les resolutions impaires produisent un decalage d'un pixel.
        """
        if x.shape[-2:] == target.shape[-2:]:
            return x
        return F.interpolate(x, size=target.shape[-2:], mode="nearest")
