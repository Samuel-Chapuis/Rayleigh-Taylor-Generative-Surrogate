# Schema: score DDPM conditionnel vs score par potentiels de Mallat

Ce fichier compare deux manieres de construire le score utilise pour generer les details d'ondelettes.

- Methode du code actuel : le score est appris implicitement via une prediction de bruit `epsilon_theta`.
- Methode Mallat : le score est construit explicitement comme combinaison de gradients de potentiels `U_m`.

## Vue Globale

```mermaid
flowchart LR
    subgraph A["Ton code: DDPM conditionnel epsilon-prediction"]
        A0["Donnees ondelettes normalisees<br/>x0 = details propres<br/>cA = approximation conditionnante"]
        A1["Forward DDPM sur les details seulement<br/>x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) epsilon<br/>epsilon ~ N(0,I)"]
        A2["Entree reseau<br/>concat(cA, x_t), t"]
        A3["U-Net conditionnel<br/>epsilon_theta(x_t,t,cA)"]
        A4["Loss d'entrainement<br/>L = E ||epsilon - epsilon_theta(x_t,t,cA)||^2"]
        A5["Score implicite<br/>s_theta(x_t,t,cA) ~= - epsilon_theta(x_t,t,cA) / sqrt(1-alpha_bar_t)"]
        A6["Sampling reverse DDPM<br/>x_{t-1} = f(x_t, epsilon_theta, beta_t) + bruit"]
        A7["Details generes<br/>p_theta(details | cA)"]

        A0 --> A1 --> A2 --> A3 --> A4
        A3 --> A5 --> A6 --> A7
    end

    subgraph B["Mallat: score matching par potentiels"]
        B0["Donnees ondelettes<br/>x = coefficients ou concat(cA, details)"]
        B1["Choix d'une famille de potentiels scalaires<br/>U_m(x) = sum_r rho_m(P x)_r<br/>P = identite, reconstruction, decomposition, etc."]
        B2["Calcul analytique des gradients<br/>grad U_m(x)<br/>champ vectoriel dans l'espace des variables generees"]
        B3["Calcul analytique des laplaciens<br/>Delta U_m(x) = div grad U_m(x)"]
        B4["Score matching de Hyvarinen<br/>A_mn = E <grad U_m, grad U_n><br/>b_m = E Delta U_m"]
        B5["Resolution des coefficients<br/>A theta = b"]
        B6["Score explicite contraint<br/>s_theta(x) = - sum_m theta_m grad U_m(x)"]
        B7["Reverse SDE / Langevin<br/>dx = drift(s_theta, x) dt + bruit"]
        B8["Echantillons generes<br/>details ou coefficients ondelettes"]

        B0 --> B1
        B1 --> B2
        B1 --> B3
        B2 --> B4
        B3 --> B4
        B4 --> B5 --> B6 --> B7 --> B8
    end
```

## Methode 1: ton DDPM conditionnel

```mermaid
flowchart TD
    A["1. Donnees d'apprentissage<br/>coefficient complet normalise: [cA, cH, cV, cD]<br/>prior = cA<br/>x0 = [cH, cV, cD]"]
    B["2. Diffusion directe uniquement sur les details<br/>x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) epsilon<br/>epsilon ~ N(0,I)"]
    C["3. Conditionnement<br/>entree U-Net = concat(prior, x_t)<br/>donc entree = [cA, cH_t, cV_t, cD_t]"]
    D["4. Prediction de bruit<br/>epsilon_theta = U-Net(concat(cA,x_t), t)"]
    E["5. Objectif optimise<br/>L_DDPM = E_{x0,t,epsilon} ||epsilon - epsilon_theta||^2"]
    F["6. Score implicite<br/>s_theta(x_t,t | cA) ~= -epsilon_theta(x_t,t,cA) / sqrt(1-alpha_bar_t)"]
    G["7. Generation<br/>initialiser x_T ~ N(0,I)<br/>appliquer la chaine reverse DDPM conditionnee par cA"]
    H["8. Sortie<br/>details generes: [cH,cV,cD]<br/>puis IDWT avec cA"]

    A --> B --> C --> D --> E
    D --> F --> G --> H
```

Description des etapes :

1. `cA` est conserve comme variable conditionnante. Les details `x0 = [cH,cV,cD]` sont les variables aleatoires a generer.
2. Le processus forward ajoute du bruit gaussien aux details, pas au prior.
3. Le reseau recoit `cA` et les details bruites par concatenation de canaux.
4. Le U-Net ne predit pas directement le score. Il predit le bruit `epsilon`.
5. La loss est une MSE de bruit. Elle favorise une bonne prediction moyenne de `epsilon`.
6. Le score est reconstruit implicitement par la relation DDPM `score ~= -epsilon_theta / sigma_t`.
7. Le sampling utilise ce score implicite dans la formule reverse.
8. Les details generes sont recombines avec `cA` par transformee ondelette inverse.

Point structurel :

```math
s_\theta(x_t,t \mid cA)
\approx
\nabla_{x_t} \log p_t(x_t \mid cA)
\approx
-\frac{\epsilon_\theta(x_t,t,cA)}{\sqrt{1-\bar\alpha_t}}
```

Le score est donc un champ vectoriel neuronal libre. Sa regularite depend du U-Net, de la normalisation, du conditionnement et des donnees.

## Methode 2: Mallat par potentiels

```mermaid
flowchart TD
    A["1. Donnees dans l'espace ondelette<br/>x = coefficients ondelettes<br/>cas conditionnel: x = concat(cA, details)"]
    B["2. Definition de potentiels scalaires<br/>U_m(x) = sum_r rho_m(P x)_r<br/>P peut etre identite, reconstruction ondelette, decomposition, etc."]
    C["3. Gradient analytique<br/>grad U_m(x)<br/>direction dans laquelle x modifie le potentiel U_m"]
    D["4. Laplacien analytique<br/>Delta U_m(x) = div grad U_m(x)<br/>terme necessaire au score matching"]
    E["5. Parametrisation du score<br/>s_theta(x) = - sum_m theta_m grad U_m(x)"]
    F["6. Score matching<br/>L(theta) = 1/2 E ||s_theta(x)||^2 + E div s_theta(x)"]
    G["7. Systeme quadratique<br/>A_mn = E <grad U_m, grad U_n><br/>b_m = E Delta U_m<br/>A theta = b"]
    H["8. Sampling reverse SDE / Langevin<br/>drift construit avec s_theta(x)<br/>ajout de bruit brownien"]
    I["9. Sortie<br/>coefficients/details generes"]

    A --> B --> C
    B --> D
    C --> E
    C --> G
    D --> G
    E --> F
    F --> G --> H --> I
```

Description des etapes :

1. Les donnees sont vues comme des champs de coefficients ondelettes.
2. Les `U_m` sont des fonctions scalaires qui mesurent des statistiques locales ou inter-echelles du champ.
3. `grad U_m(x)` est un champ vectoriel. C'est une brique elementaire du score.
4. `Delta U_m(x)` est calcule analytiquement. Il sert a evaluer la divergence du score dans la loss de Hyvarinen.
5. Le score n'est pas un reseau libre. Il est contraint a etre une combinaison lineaire de gradients de potentiels.
6. Le score matching evite d'avoir besoin du vrai score des donnees.
7. Comme la parametrisation est lineaire en `theta`, l'optimisation se reduit a un probleme quadratique.
8. Le score explicite est injecte dans une dynamique reverse SDE ou dans des corrections de Langevin.
9. Les echantillons suivent les contraintes statistiques imposees par les potentiels choisis.

Point structurel :

```math
E_\theta(x) = \sum_m \theta_m U_m(x)
```

```math
s_\theta(x)
= \nabla_x \log p_\theta(x)
\approx -\nabla_x E_\theta(x)
= -\sum_m \theta_m \nabla_x U_m(x)
```

Score matching de Hyvarinen :

```math
\mathcal L(\theta)
=
\frac{1}{2}\mathbb E_{p_{data}} \|s_\theta(x)\|^2
+
\mathbb E_{p_{data}}[\nabla \cdot s_\theta(x)]
```

Avec `s_theta(x) = -sum_m theta_m grad U_m(x)`, on obtient :

```math
A_{mn}
=
\mathbb E
\left[
\langle \nabla U_m(x), \nabla U_n(x) \rangle
\right]
```

```math
b_m
=
\mathbb E
\left[
\Delta U_m(x)
\right]
```

```math
A\theta = b
```

## Exemple concret de potentiel

Potentiel local quartique simple :

```math
U(x) = \sum_r x_r^4
```

Gradient :

```math
\nabla U(x)_r = 4 x_r^3
```

Interprétation :

- `U(x)` mesure l'energie non gaussienne des grandes amplitudes.
- `grad U(x)` produit une force tres forte sur les coefficients de grande amplitude.
- Cette force peut aider a modeliser des queues de distribution et de l'intermittence.

Potentiel applique apres reconstruction ondelette :

```math
U_m(w)
=
\sum_r \rho_m((R w)_r)
```

avec :

- `w` : coefficients ondelettes.
- `R` : reconstruction ondelette vers une echelle plus fine.
- `rho_m` : non-linearite scalaire.

Gradient :

```math
\nabla_w U_m(w)
=
R^T \rho_m'(R w)
```

Dans le code de Mallat, ce gradient est calcule par reconstruction puis redecomposition ondelette. C'est ce qui donne un score coherent avec la geometrie multi-echelle.

## Difference essentielle

| Aspect | Ton DDPM conditionnel | Mallat |
|---|---|---|
| Objet appris | Bruit `epsilon_theta` | Coefficients `theta_m` |
| Score | Implicite, derive de `epsilon_theta` | Explicite, somme de gradients |
| Forme du score | Champ neuronal libre | Champ contraint par potentiels |
| Objectif | MSE sur le bruit | Score matching de Hyvarinen |
| Geometrie ondelette | Donnee au reseau via les canaux | Encodee dans les potentiels |
| Localite/stationnarite | Depend de l'architecture | Imposee par construction |
| Risque principal | Score moyen lisse, sous-dispersion des details | Biais fort si les potentiels sont mal choisis |

## Lecture courte

Ton modele apprend :

```math
[cA, x_t, t] \mapsto \epsilon_\theta
```

puis en deduit :

```math
s_\theta \approx -\epsilon_\theta / \sigma_t
```

Mallat construit :

```math
U_1, \dots, U_M
```

puis apprend :

```math
s_\theta(x) = -\sum_m \theta_m \nabla U_m(x)
```

La difference fondamentale est donc que Mallat choisit d'abord une famille de forces admissibles `grad U_m`, puis apprend leur combinaison. Ton DDPM apprend directement une force generale via un reseau.


$$
x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon
$$

$$
\epsilon_\theta(x_t,t,cA)
$$

$$
dX_t = -X_t dt + \sqrt{2\sigma}\,dB_t
$$