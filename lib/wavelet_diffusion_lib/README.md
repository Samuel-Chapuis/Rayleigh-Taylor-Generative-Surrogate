# wavelet_diffusion_lib

Composants strictement spécifiques à la cascade ondelette :

- `ConditionalDDPM.py` : diffusion des détails conditionnée par `cA`;
- `ConditionalSGM.py` : VP-SGM conditionnel des détails;
- `training_loop.py` : losses et boucles conditionnelles;
- `wavelet_utils.py` : tenseurs et pré/post-traitements de coefficients.

Les modèles génériques, U-Net, schedules, visualisation, journalisation,
gestion du device et loaders communs sont dans `lib.diffusion_lib`.
