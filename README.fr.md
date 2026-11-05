<div align="center">

# pyfai-zero-shot

**Calibration géométrique zero-shot et aveugle en phase de détecteurs RX 2D<br>à partir d'une seule image d'anneaux de poudre**

🚫 pas de phase-étalon connue &nbsp;·&nbsp; 🚫 pas de géométrie initiale &nbsp;·&nbsp; 🚫 longueur d'onde optionnellement inconnue<br>
✅ sortie en PONI pyFAI standard &nbsp;·&nbsp; ✅ incertitudes bootstrap

[English](README.md) · [简体中文](README.zh-CN.md) · [正體中文](README.zh-TW.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · **Français**

[![License: MIT](https://img.shields.io/badge/licence-MIT-1a365d)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.5_préprint-1a365d)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![Built for pyFAI](https://img.shields.io/badge/pour-pyFAI-013243)]()
[![Statut](https://img.shields.io/badge/statut-manuscrit_en_préparation-e67e22)]()

</div>

---

## Pourquoi

Les outils de calibration classiques (pyFAI `calib2`, DAWN, FIT2D) supposent
une **phase-étalon connue, une sélection manuelle des anneaux et une assez
bonne géométrie initiale**. Lancés à froid, ils se verrouillent sur des
solutions auto-cohérentes mais *fausses* — la dégénérescence indexation des
anneaux / distance — avec un résidu faible qui masque l'erreur. Sur le
benchmark public ci-dessous, pyFAI à froid échoue sur **11/11** images ;
cette pipeline résout la plupart d'entre elles sans supervision, à partir
d'une seule image, sans qu'on lui dise ce qu'est l'échantillon.

## Fonctionnement

Une **cascade classique d'abord** sur une bibliothèque de phases de poudre,
arbitrée par une **porte de consensus entre germes** (cross-seed consensus) :

1. un solveur rapide de positions de pics démarre depuis le centre du
   canevas (15 ms/image) ;
2. en cas d'échec seulement, des germes de repli entrent en jeu — un
   estimateur fermé du centre par symétrie radiale, des extrapolations de
   rayons compensant son biais hors axe, et optionnellement un réseau de
   neurones en régression ;
3. les germes votent ; la solution que des germes indépendants retrouvent
   l'emporte, éliminant les pseudo-solutions ;
4. affinage des inclinaisons + rééchantillonnage bootstrap des photons →
   un **PONI** standard avec incertitudes par grandeur.

<div align="center">
<img src="assets/fig1_architecture.png" alt="Architecture de la pipeline" width="820">
</div>

## Résultats en un coup d'œil

| Benchmark | Résultat |
|---|---|
| Étalon CeO₂ · 7 distances, 150–450 mm | \|ΔD\| moyenne et pire = **0,1 mm**, 7/7 convergés ; chemin classique 15 ms/image |
| Al₂O₃ · 52 images statiques | 51/52 résolues (une image sombre correctement rejetée) ; dispersion **MAD 0,45 mm** ; bootstrap σ(D) = **0,47 mm**, couverture vraie valeur 6/6 |
| Inter-instruments · 10 images publiques (8 évaluables) | phase identifiée **6/8** ; \|ΔD\| ≤ 0,1 mm sur **4/8** ; inclinaisons ≲ 5 mrad ; *chaque échec est expliqué* |
| Vérification indépendante | pyFAI à froid échoue sur les **11/11** mêmes images ; parti de notre solution, il nous rejoint — le gain vient de l'initialisation, pas d'un autre moteur |

<div align="center">
<img src="assets/fig2_frames_rings.png" alt="Images du benchmark avec anneaux ajustés" width="760"><br>
<sub>Images du benchmark avec anneaux ajustés : (a) CeO₂ à 300 mm (8 anneaux appariés) ; (b) moyenne des 52 images statiques Al₂O₃.</sub>
</div>

<details>
<summary><b>Plus de preuves</b> (erreurs de distance CeO₂ · duel indépendant avec pyFAI)</summary>

<div align="center">
<img src="assets/fig3_ceria_errors.png" width="520"><br>
<sub>Erreurs de CeO₂ sur sept distances (affinage 2D natif) : |ΔD| moyenne et pire = 0,1 mm.</sub><br><br>
<img src="assets/fig8_duel_ablation.png" width="760"><br>
<sub>Comparaison indépendante avec pyFAI (départ à froid P0 contre cette pipeline T, départ identique P1) et ablation des composants.</sub>
</div>

</details>

## Démarrage rapide

```bash
pip install numpy scipy matplotlib pyFAI fabio pillow h5py torch
```

```python
import fabio
from probes.solver_core import solve_image

img = fabio.open("votre_image.edf").data.astype(float)
img[img >= 1e9] = 0.0            # masquage sentinelles/négatifs (schéma zoo)
img[img < 0] = 0.0

res = solve_image(img, px_m=172e-6)      # lam=None → mode λ–D conjoint
print(res["phase"], res["sol"])          # phase identifiée + solution géométrique
```

Téléchargez les images de test publiques référencées dans l'article
(<https://www.silx.org/pub/pyFAI/testimages/>) dans `data/real/`, puis
lancez les sondes — chaque tableau de `probes/results/` est régénéré de
manière déterministe (graines fixées).

| Point d'entrée | Rôle |
|---|---|
| `probes/solver_core.py` | entrée unifiée `solve_image()` (canevas + repli sûr en zero-shot) |
| `probes/zoo_run.py` | benchmark inter-instruments (images de test silx) |
| `probes/phase_map.py` | carte des conditions (exposition / inclinaison / extension) |
| `probes/ablation8.py` | ablation des composants (jeu stressant) |
| `probes/_pyfai_duel.py` | comparaison indépendante avec pyFAI |
| `generators/train.py` | entraînement de l'initialiseur NN (images synthétiques) |

## Organisation du dépôt

```
torchfai/      bibliothèque de solveur (phases, simulateur, affinage, modèle)
probes/        sondes de benchmark, scripts de figures, solveur graphique
  results/     tableaux de résultats complets (.md/.json/.txt) + PONI de référence
generators/    générateurs de données d'entraînement NN + outils de fusion
docs/          documents techniques de preuve (journaux d'expérience)
manuscript/    projet d'article v0.5 (Markdown)
assets/        figures utilisées dans ce README
```

## Invitation au test à l'aveugle

> Nous collectons **1–2 images historiques + leur .poni de référence** sur
> n'importe quelle ligne de faisceau — tout détecteur, étalon, distance,
> longueur d'onde, époque. Vous envoyez l'image et la calibration connue ;
> nous résolvons totalement à l'aveugle et renvoyons ΔD, centre du faisceau,
> inclinaisons, un PONI prêt à l'emploi, des intervalles d'incertitude et
> l'identification de la phase. Nous ne voyons la référence qu'après que
> notre solution est figée et horodatée. **Les résultats négatifs sont
> rapportés tels quels** — chaque échec est attribué, jamais excusé.

**Contact :** Ziao Liu · ziaoliu902@gmail.com

## Citation

```bibtex
@software{pyfai_zero_shot_2026,
  author = {Liu, Ziao},
  title  = {pyfai-zero-shot: phase-blind, zero-shot geometry calibration
            of 2-D X-ray detectors from powder rings},
  year   = {2026},
  url    = {https://github.com/ziaoliu902-cyber/pyfai-zero-shot},
  note   = {v0.5}
}
```

## Licence et remerciements

MIT (voir [LICENSE](LICENSE)). Les images du benchmark sont les images de
test pyFAI diffusées publiquement (silx.org, acquises à D2AM/BM02, ESRF) ;
ce projet est un utilisateur indépendant de pyFAI, sans lien avec l'équipe
pyFAI.

<sub>Le manuscrit (v0.5) est un projet en cours de rédaction et n'a pas
été soumis à la révision par les pairs. Les README multilingues sont
maintenus à la main ; la version anglaise fait foi.</sub>
