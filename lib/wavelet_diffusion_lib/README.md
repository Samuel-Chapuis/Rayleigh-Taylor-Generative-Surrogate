# wavelet_diffusion_lib

Ce document regroupe les docstrings presentes dans les modules de `wavelet_diffusion_lib`.
Il est organise par fichier, puis par classe, methode ou fonction.

## `Attention.py`

### `class AttentionBlock`

An attention block that allows spatial positions to attend to each other.

Originally ported from here, but adapted to the N-d case.
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.

#### `AttentionBlock.__init__`

Initialise le bloc d'attention spatiale.

Args:
    channels: Nombre de canaux du tenseur d'entree.
    num_heads: Nombre de tetes d'attention.
    use_checkpoint: Active le checkpointing gradient pour reduire la memoire.

#### `AttentionBlock.forward`

Applique l'attention avec checkpointing optionnel.

Args:
    x: Tenseur de forme ``(N, C, *spatial)``.

Returns:
    Tenseur de meme forme que ``x``.

#### `AttentionBlock._forward`

Calcule l'attention en aplatissant les dimensions spatiales.

Les positions spatiales deviennent la dimension de sequence ``T`` pour
l'attention QKV, puis la forme spatiale initiale est restauree.

### `class QKVAttention`

A module which performs QKV attention.

#### `QKVAttention.forward`

Apply QKV attention.

:param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
:return: an [N x C x T] tensor after attention.

#### `QKVAttention.count_flops`

A counter for the `thop` package to count the operations in an
attention operation.

Meant to be used like:

    macs, params = thop.profile(
        model,
        inputs=(inputs, timestamps),
        custom_ops={QKVAttention: QKVAttention.count_flops},
    )

## `ConditionalDDPM.py`

### `class WaveletConditionalDDPM`

DDPM conditionnel pour coefficients d'ondelettes.

Le canal 0 est le prior basse frequence cA. Les canaux 1..3 sont les details
cH/cV/cD sur lesquels on applique la diffusion et dont le bruit est predit.

#### `WaveletConditionalDDPM.__init__`

Initialise un DDPM conditionnel sur un prior wavelet basse frequence.

Args:
    network: Reseau qui predit le bruit sur les canaux de details.
    n_steps: Nombre de pas du processus de diffusion.
    min_beta: Valeur initiale du calendrier lineaire des variances.
    max_beta: Valeur finale du calendrier lineaire des variances.
    device: Device PyTorch utilise pour le modele et les tenseurs.
    prior_channels: Nombre de canaux conserves comme condition.
    target_channels: Nombre de canaux de details diffuses.
    image_hw: Resolution spatiale des coefficients wavelet.
    coeff_mean: Moyenne canal par canal pour denormaliser les coefficients.
    coeff_std: Ecart type canal par canal pour normaliser les coefficients.

#### `WaveletConditionalDDPM.forward`

Applique le processus direct uniquement aux canaux de details.

Args:
    details_0: Details propres de forme ``(N, target_channels, H, W)``.
    t: Pas de diffusion pour chaque element du batch.
    eta: Bruit gaussien optionnel. Si ``None``, il est echantillonne.

Returns:
    Details bruites au pas ``t``.

#### `WaveletConditionalDDPM.backward`

Predit le bruit des details conditionnellement au prior.

Args:
    noisy_details: Details bruites de forme ``(N, target_channels, H, W)``.
    t: Pas de diffusion associes au batch.
    prior: Canaux conditionnants, typiquement ``cA``.

Returns:
    Estimation du bruit ajoute aux details.

#### `WaveletConditionalDDPM.sample`

Genere des details par diffusion inverse conditionnee par le prior.

Args:
    prior: Tenseur conditionnant de forme ``(N, prior_channels, H, W)``.
    device: Device optionnel pour l'echantillonnage.

Returns:
    Coefficients concatenes ``(prior, details_generes)``.

#### `WaveletConditionalDDPM.denormalize_coeffs`

Repasse des coefficients normalises vers l'echelle physique/statistique.

#### `WaveletConditionalDDPM.normalize_coeffs`

Normalise des coefficients avec les statistiques stockees dans le modele.

#### `WaveletConditionalDDPM._flatten_time`

Convertit un pas de temps scalaire ou tensoriel en vecteur ``long``.

## `data_loader.py`

### `class ProcessedDataset`

Jeu de données PyTorch basé sur des fichiers déjà prétraités.

Les images sont chargées depuis un fichier ``.pt`` contenant les tenseurs
d'images et d'étiquettes, puis normalisées dans l'intervalle ``[-1, 1]``.

#### `ProcessedDataset.__init__`

Charge le jeu de données prétraité.

Args:
    root (str): Répertoire racine du jeu de données.
    train (bool | None, optional): Ancienne API. Si True, charge
        ``training``; si False, charge ``test``.
    split (str | None, optional): Partition explicite à charger parmi
        ``training``, ``validation`` ou ``test``. Si fourni, remplace
        ``train``.

Raises:
    FileNotFoundError: Si le fichier ``training.pt`` ou ``test.pt`` est
        introuvable.

#### `ProcessedDataset.__len__`

Retourne le nombre d'exemples disponibles.

Returns:
    int: Nombre d'éléments du jeu de données.

#### `ProcessedDataset.__getitem__`

Récupère un exemple du jeu de données.

Args:
    index (int): Indice de l'exemple à récupérer.

Returns:
    tuple[torch.Tensor, torch.Tensor]: Une image normalisée et son label.

### `data_loader`

Construit un DataLoader pour une partition prétraitée.

Args:
    config: Objet de configuration contenant au moins ``store_path_dataset``
        et ``batch_size``.

Returns:
    torch.utils.data.DataLoader: Chargeur de données prêt pour l'entraînement.

## `DDPM.py`

### `class DDPM`

Modèle de diffusion probabiliste pour la génération d'images.

Cette classe encapsule le processus direct de diffusion, le processus inverse
appris par le réseau de neurones, ainsi que l'échantillonnage d'images à partir
d'un bruit gaussien.

#### `DDPM.__init__`

Initialise un modèle DDPM.

Args:
    network (torch.nn.Module): Réseau chargé de prédire le bruit à partir
        d'une image bruitée et d'un pas de temps.
    n_steps (int, optional): Nombre total d'étapes de diffusion. Par défaut 200.
    min_beta (float, optional): Valeur minimale du calendrier de bruit.
        Par défaut 10**-4.
    max_beta (float, optional): Valeur maximale du calendrier de bruit.
        Par défaut 0.02.
    device (torch.device | None, optional): Dispositif de calcul à utiliser.
        Par défaut None.
    image_chw (tuple[int, int, int], optional): Format des images sous la forme
        (canaux, hauteur, largeur). Par défaut (1, 28, 28).

#### `DDPM.forward`

Applique le processus direct de diffusion.

Cette méthode ajoute du bruit à une image propre `x0` jusqu'au pas `t`.

Args:
    x0 (torch.Tensor): Images propres de forme ``(N, C, H, W)``.
    t (torch.Tensor): Pas de temps pour chaque élément du lot.
    eta (torch.Tensor | None, optional): Bruit à injecter. Si `None`, un
        bruit gaussien est généré automatiquement.

Returns:
    torch.Tensor: Images bruitées de même forme que `x0`.

#### `DDPM.backward`

Applique le processus inverse appris par le réseau.

Args:
    x (torch.Tensor): Images bruitées.
    t (torch.Tensor): Pas de temps associé à chaque image.

Returns:
    torch.Tensor: Estimation du bruit prédit par le réseau.

#### `DDPM.sample`

Génère des images en partant d'un bruit gaussien.

Args:
    n_samples (int, optional): Nombre d'images à générer. Par défaut 16.
    device (torch.device | None, optional): Dispositif de calcul à utiliser.
        Par défaut None.
    c (int | None, optional): Nombre de canaux des images générées.
        Par défaut la première valeur de `image_chw`.
    h (int | None, optional): Hauteur des images générées. Par défaut la
        deuxième valeur de `image_chw`.
    w (int | None, optional): Largeur des images générées. Par défaut la
        troisième valeur de `image_chw`.

Returns:
    torch.Tensor: Lot d'images générées.

## `embeding.py`

### `sinusoidal_embedding`

Construit l'embedding positionnel sinusoidal standard.

Args:
    n: Nombre de positions ou pas de temps.
    d: Dimension de l'embedding.

Returns:
    Tenseur de forme ``(n, d)`` alternant sinus et cosinus a plusieurs
    frequences.

## `ImageVisualizer.py`

### `class ImageVisualizer`

Utilitaire de visualisation pour afficher des lots d'images et suivre la
diffusion directe et inverse.

#### `ImageVisualizer.__init__`

Initialise le visualiseur.

Args:
    figsize (tuple, optional): Taille de la figure Matplotlib utilisée pour l'affichage.
    cmap (str, optional): Palette utilisée pour les images en niveaux de gris.
    output_dir (str | None, optional): Dossier où sauvegarder les figures générées.

#### `ImageVisualizer._save_figure`

Sauvegarde une figure si un répertoire de sortie a été configuré.

Args:
    fig (matplotlib.figure.Figure): Figure à sauvegarder.
    filename (str): Nom du fichier de sortie.

#### `ImageVisualizer._to_numpy`

Convertit un tenseur PyTorch en tableau NumPy si nécessaire.

Args:
    images: Tenseur PyTorch ou tableau déjà compatible NumPy.

Returns:
    Les données sous forme de tableau NumPy, ou l'entrée d'origine si elle l'est déjà.

#### `ImageVisualizer.show_images`

Affiche un lot d'images dans une grille.

Args:
    images: Lot d'images au format tenseur ou NumPy, attendu sous la
        forme ``(N, C, H, W)``.
    title (str, optional): Titre affiché en haut de la figure.

#### `ImageVisualizer.show_first_batch`

Affiche le premier lot produit par un DataLoader.

Args:
    loader: Itérateur PyTorch fournissant des lots sous la forme
        ``(images, labels, ...)``.

#### `ImageVisualizer.show_forward`

Affiche des images bruitées à plusieurs niveaux du processus direct.

Args:
    ddpm: Modèle DDPM fournissant le processus de diffusion directe.
    loader: DataLoader contenant des images propres.
    device: Périphérique sur lequel exécuter les calculs.

#### `ImageVisualizer.show_backward`

Affiche l'évolution de la génération pendant le processus inverse.

Args:
    ddpm: Modèle DDPM utilisé pour la génération.
    device: Périphérique sur lequel générer les images.
    n_samples (int, optional): Nombre d'images affichées par ligne.

## `Logger.py`

### `class Logger`

Logger d'experience ecrivant a la fois un fichier texte et un CSV de metriques.

Le fichier texte conserve les messages chronologiques. Le CSV stocke les
resumes d'epoque et peut etendre automatiquement son en-tete si de nouvelles
metriques apparaissent.

#### `Logger.__init__`

Initialise les fichiers de log et configure le logger Python.

Args:
        log_path: Chemin du fichier texte.
        csv_path: Chemin du fichier CSV. Si ``None``, remplace l'extension de
                ``log_path`` par ``.csv``.
        name: Nom du logger Python sous-jacent.

#### `Logger.info`

Ecrit un message de niveau info dans le log texte.

#### `Logger.warning`

Ecrit un message de niveau warning dans le log texte.

#### `Logger.error`

Ecrit un message de niveau error dans le log texte.

#### `Logger.log_experiment_start`

Journalise le debut d'une experience et ses parametres.

#### `Logger.save_config`

Sauvegarde une configuration JSON serialisable.

#### `Logger.log_epoch`

Ajoute les metriques d'une epoque au CSV et au log texte.

#### `Logger.log_experiment_end`

Journalise la fin d'une experience.

#### `Logger._append_csv_row`

Ajoute une ligne au CSV en etendant l'en-tete si necessaire.

#### `Logger._read_csv_header`

Lit l'en-tete du CSV existant, s'il existe.

#### `Logger._rewrite_csv_with_header`

Reecrit le CSV avec un nouvel ordre de colonnes.

#### `Logger._json_ready`

Convertit recursivement une valeur en structure compatible JSON.

## `schedules.py`

### `linear_beta_schedule`

Calendrier lineaire classique des variances beta_t.

### `snr_from_alpha_bar`

Calcule SNR(t) = alpha_bar(t) / (1 - alpha_bar(t)).

### `diffusion_steps_from_snr`

Calcule le nombre de pas a conserver a partir d'un seuil SNR.

Convention utilisee dans le notebook 1D: on construit un calendrier lineaire
de reference, puis on garde le premier prefixe tel que
``SNR(t) <= snr_threshold``.

### `get_diffusion_schedule`

Retourne les buffers beta/alpha/alpha_bar sur le device et dtype demandes.

Si un modele DDPM est fourni, ses buffers sont reutilises exactement. Sinon,
un calendrier lineaire est reconstruit.

## `training_loop.py`

### `_ddpm_noise_prediction_loss`

Calcule la loss DDPM epsilon-prediction sur un batch.

### `evaluate_loss`

Evalue la loss moyenne d'un DDPM sans mettre a jour les poids.

### `training_loop`

Entraîne un modèle DDPM sur un jeu de données.

Args:
    ddpm (DDPM): Modèle de diffusion à entraîner.
    loader (torch.utils.data.DataLoader): Chargeur fournissant les lots
        d'images d'entraînement.
    n_epochs (int): Nombre d'époques d'entraînement.
    optim (torch.optim.Optimizer): Optimiseur utilisé pour la mise à jour
        des paramètres.
    device (torch.device): Dispositif de calcul utilisé pendant l'entraînement.
    display (ImageVisualizer | None, optional): Visualiseur optionnel pour
        afficher des échantillons générés en fin d'époque.
    store_path (str, optional): Chemin de sauvegarde du meilleur modèle.

Args:
    logger (Logger | None, optional): Logger optionnel pour écrire le déroulé
        de l'expérience et les métriques d'époque.
    val_loader (torch.utils.data.DataLoader | None, optional): Chargeur de
        validation. Si fourni, le meilleur modèle est sauvegardé sur la
        loss de validation plutôt que sur la loss d'entraînement.

Returns:
    list[float]: Historique des pertes calculées à chaque itération.

### `split_wave_batch`

Separe un batch de coefficients wavelet en prior et details.

Args:
    batch: Batch issu d'un ``TensorDataset`` contenant un tenseur
        ``(N, C, H, W)`` en premiere position.
    device: Device vers lequel deplacer les coefficients.
    prior_channels: Nombre de premiers canaux utilises comme condition.

Returns:
    Tuple ``(prior, details)``. Pour le cas wavelet standard, ``prior`` est
    ``cA`` et ``details`` contient ``cH/cV/cD``.

### `wave_noise_prediction_loss`

Calcule la loss epsilon-prediction du DDPM conditionnel wavelet.

Le bruit est ajoute uniquement aux canaux de details. Le prior est concatene
aux details bruites dans ``ddpm.backward`` pour conditionner la prediction.

### `wave_evaluate_loss`

Evalue la loss moyenne du DDPM conditionnel wavelet sans mise a jour.

Le mode train/eval initial du modele est restaure apres l'evaluation.

### `wave_training_loop`

Entraine un DDPM conditionnel sur coefficients wavelet.

Le modele apprend a predire le bruit sur les details conditionnellement au
prior basse frequence. Le meilleur checkpoint est selectionne sur la loss de
validation si ``val_loader`` est fourni, sinon sur la loss d'entrainement.

Args:
    ddpm: Instance de ``WaveletConditionalDDPM``.
    loader: DataLoader d'entrainement.
    n_epochs: Nombre d'epoques.
    optim: Optimiseur PyTorch.
    device: Device de calcul.
    store_path: Chemin de sauvegarde du meilleur checkpoint.
    logger: Logger optionnel.
    val_loader: DataLoader de validation optionnel.

Returns:
    Historique des pertes batch par batch.

## `UNet.py`

### `class Block`

Bloc convolutionnel utilisé dans le U-Net.

Le bloc applique éventuellement une normalisation par couche, puis deux
convolutions successives séparées par une activation.

#### `Block.__init__`

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

#### `Block.forward`

Propulse les données à travers le bloc convolutionnel.

Args:
    x (torch.Tensor): Tenseur d'entrée.

Returns:
    torch.Tensor: Tenseur transformé par le bloc.

### `class UNet`

Architecture U-Net conditionnée par le temps.

Le réseau reçoit une image bruitée et un pas de diffusion, puis prédit le
bruit associé à cette image.

#### `UNet.__init__`

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

#### `UNet.forward`

Prédit le bruit associé à une image bruitée.

Args:
    x (torch.Tensor): Images d'entrée de forme ``(N, C, H, W)``.
    t (torch.Tensor): Pas de temps pour chaque image du lot.

Returns:
    torch.Tensor: Carte de bruit prédite.

#### `UNet._make_block_stack`

Construit une pile de blocs convolutionnels a resolution fixe.

Args:
    size: Taille spatiale carree traitee par les blocs.
    in_channels: Nombre de canaux en entree du premier bloc.
    out_channels: Nombre de canaux en sortie de chaque bloc.
    num_blocks: Nombre de blocs a empiler.
    final_normalize: Desactive la normalisation du dernier bloc si faux.

Returns:
    Sequence PyTorch de blocs convolutionnels.

#### `UNet._make_te`

Construit un petit réseau pour projeter l'embedding temporel.

Args:
    dim_in (int): Dimension d'entrée.
    dim_out (int): Dimension de sortie.

Returns:
    torch.nn.Sequential: Bloc de projection de l'embedding temporel.

#### `UNet._conv_size`

Calcule la taille spatiale de sortie d'une convolution 2D.

La formule suit la convention PyTorch pour une dimension spatiale.

#### `UNet._resize_to`

Redimensionne ``x`` vers la resolution spatiale de ``target`` si besoin.

Ce correctif evite les erreurs de concaténation des skip connections
quand les resolutions impaires produisent un decalage d'un pixel.

## `utils.py`

### `get_best_device`

Détermine le meilleur dispositif de calcul disponible (GPU, MPS ou CPU).

La priorité est donnée à CUDA, puis à MPS, puis au CPU.

Returns:
    torch.device: le dispositif de calcul à utiliser

## `wavelet_utils.py`

### `load_wave_tensor`

Charge un tenseur wavelet deja pretraite au format (N, C, H, W).

### `channel_stats`

Statistiques canal par canal, partagees par train/validation/test.

### `normalize_with_stats`

Normalisation affine avec des statistiques de forme (C,).

### `make_loader`

Construit un DataLoader sans label: chaque batch contient seulement les coefficients.

### `show_wave_channels`

Sauvegarde une grille simple: chaque coefficient devient une image affichee.

### `build_wavelet_model`

Construit le U-Net et l'encapsule dans le DDPM conditionnel wavelet.

### `do_diffusion_until_snr`

Applique le processus direct de diffusion jusqu'a un seuil de SNR.

Par defaut, le prior basse frequence cA est conserve et seuls les canaux de
detail cH/cV/cD sont bruites. Utiliser ``diffuse_prior=True`` pour bruiter
tous les canaux.

``reference_steps`` controle le calendrier long sur lequel le seuil SNR est
evalue lorsque ``steps`` n'est pas fourni.

## `__init__.py`

Aucune classe ou fonction exposee dans ce fichier.
