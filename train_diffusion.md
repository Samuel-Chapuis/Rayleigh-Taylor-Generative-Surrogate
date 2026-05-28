# Lancement de `train_diffusion.sh`

Ce fichier documente le script de lancement utilisé sur TGCC pour entraîner le modèle de diffusion du projet.

## Rôle du script

`train_diffusion.sh` soumet un job batch via `MSUB`, charge les modules nécessaires, active l'environnement Python du projet puis lance `diffusion.py`.

Le script est prévu pour une exécution sur une machine ou un nœud TGCC disposant de `module`, de `MSUB` et d'un accès au répertoire de travail fourni par l'environnement (`$CCCWORKDIR`).

## Prérequis

Avant de lancer le script, vérifier que :

- l'environnement virtuel Python existe bien à l'emplacement `../.venv` par rapport au dossier `diffusion/`
- le répertoire courant du projet est bien accessible depuis `$CCCWORKDIR/diffusion`
- les dossiers `logs/` et `outputs/` existent si vous souhaitez conserver les sorties et erreurs du job
- les modules `python3` et `cuda` sont disponibles sur la plateforme TGCC utilisée

## Directives TGCC `MSUB`

Les lignes commençant par `#MSUB` sont des directives de soumission de job.

- `#MSUB -r diffusion_train` : nom du job affiché par le scheduler
- `#MSUB -q a100` : file d'attente cible, ici une file adaptée aux GPU A100
- `#MSUB -n 1` : nombre de processus demandés
- `#MSUB -c 64` : nombre de cœurs CPU demandés, pas de GPU
- `#MSUB -T 86400` : durée maximale du job en secondes, ici 24 heures
- `#MSUB -m work,scratch` : montage des espaces `work` et `scratch`
- `#MSUB -o logs/out_%I.txt` : fichier de sortie standard
- `#MSUB -e logs/err_%I.txt` : fichier d'erreur standard

Le suffixe `%I` permet de différencier les fichiers si le job génère plusieurs tâches ou itérations prises en charge par le scheduler.

Pour demander un nombre précis de GPU, ce n'est généralement pas `-c` qu'il faut utiliser. Sur TGCC, cette réservation passe par la file de soumission choisie ou par une option GPU spécifique du scheduler sur le site concerné. Avec ce script, la présence d'un GPU est donc implicite via la file `a100`, mais le nombre exact de GPU n'est pas fixé dans le fichier tel qu'il est écrit.

## Étapes exécutées par le script

Le script fait ensuite, dans l'ordre :

1. active l'affichage des commandes avec `set -x`
2. se place dans le répertoire du projet avec `cd $CCCWORKDIR/diffusion`
3. purge les modules chargés avec `module purge`
4. charge `python3`
5. charge `cuda`
6. active l'environnement virtuel avec `source ../.venv/bin/activate`
7. affiche l'état GPU avec `nvidia-smi`
8. lance l'entraînement ou l'exécution principale avec `python diffusion.py`

## Deux façons de lancer le script

### 1. Soumission batch TGCC

C'est la façon attendue sur la plateforme TGCC :

```bash
msub train_diffusion.sh
```

ou, selon la configuration locale du cluster :

```bash
qsub train_diffusion.sh
```

si l'environnement utilise une couche de compatibilité `MSUB`/`qsub`.

### 2. Exécution manuelle pour test local

Pour un test rapide hors scheduler, il faut adapter le script ou reproduire manuellement les étapes :

```bash
module purge
module load python3
module load cuda
source ../.venv/bin/activate
python diffusion.py
```

Cette exécution ne remplace pas le mode batch si le script doit utiliser un GPU réservé sur TGCC.

## Points à adapter selon le site

Selon la configuration du cluster, vous devrez peut-être ajuster :

- le nom de la file `a100`
- le nombre de cœurs `64`
- la durée `86400`
- les chemins `logs/`
- le chemin vers l'environnement virtuel `../.venv`
- le chemin du projet dans `cd $CCCWORKDIR/diffusion`

## Remarque

Ce script ne prend pas d'arguments en ligne de commande. Tous les paramètres de lancement sont codés directement dans le fichier.
