<div align="center">

# pyfai-zero-shot

**Phasenblinde Zero-Shot-Geometriekalibrierung von 2D-Röntgendetektoren<br>aus einem einzelnen Pulverringbild**

🚫 keine bekannte Kalibrierphase &nbsp;·&nbsp; 🚫 kein Startwert für die Geometrie &nbsp;·&nbsp; 🚫 Wellenlänge optional unbekannt<br>
✅ Ausgabe als Standard-pyFAI-PONI &nbsp;·&nbsp; ✅ Bootstrap-Unsicherheiten

[English](README.md) · [简体中文](README.zh-CN.md) · [正體中文](README.zh-TW.md) · **Deutsch** · [Русский](README.ru.md) · [Français](README.fr.md)

[![License: MIT](https://img.shields.io/badge/Lizenz-MIT-1a365d)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v0.5_Preprint-1a365d)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![Built for pyFAI](https://img.shields.io/badge/für-pyFAI-013243)]()
[![Status](https://img.shields.io/badge/Status-Manuskript_in_Arbeit-e67e22)]()

</div>

---

## Warum

Herkömmliche Kalibrierwerkzeuge (pyFAI `calib2`, DAWN, FIT2D) setzen eine
**bekannte Kalibrierphase, manuelles Ring-Picking und einen ausreichend
guten Startwert** voraus. Beim Kaltstart verriegeln sie sich in
selbstkonsistenten, aber *falschen* Lösungen — der Ringindex/Abstand-
Entartung — mit einem kleinen Residuum, das den Fehler verbirgt. Auf dem
unten genannten öffentlichen Benchmark scheitert pyFAI im Kaltstart an
**11/11** Bildern; diese Pipeline löst die meisten davon unbeaufsichtigt,
aus einem einzelnen Bild, ohne dass die Probe bekannt sein muss.

## Funktionsweise

Eine **klassisch-first Kaskade** über eine Pulverphasenbibliothek,
arbitriert durch ein **konsensbasiertes Seed-Gate** (cross-seed
consensus gate):

1. ein schneller Peak-Positions-Löser startet zuerst aus der
   Bildmitte (15 ms/Bild);
2. nur bei Fehler greifen Fallback-Seeds — ein geschlossener
   radialsymmetrischer Schätzer für das Strahlzentrum, Strahl-Extrapolationen
   zur Kompensation seines Off-Axis-Bias und optional ein Regressions-
   neuronales Netz;
3. die Seeds stimmen ab; die Lösung gewinnt, die unabhängige Seeds erneut
   finden — Pseudolösungen werden eliminiert;
4. Neigungskorrektur + Bootstrap-Photonenresampling → ein Standard-**PONI**
   mit Unsicherheiten pro Größe.

<div align="center">
<img src="assets/fig1_architecture.png" alt="Pipeline-Architektur" width="820">
</div>

## Ergebnisse auf einen Blick

| Benchmark | Ergebnis |
|---|---|
| CeO₂-Kalibrierphase · 7 Abstände, 150–450 mm | mittlerer & schlimmster \|ΔD\| = **0,1 mm**, 7/7 konvergent; klassischer Pfad 15 ms/Bild |
| Al₂O₃ · 52 statische Bilder | 51/52 gelöst (ein Dunkelbild korrekt verworfen); Streuung **MAD 0,45 mm**; Bootstrap σ(D) = **0,47 mm**, Ground-Truth-Coverage 6/6 |
| Gerätecbergreifend · 10 öffentliche Bilder (8 auswertbar) | Phase identifiziert **6/8**; \|ΔD\| ≤ 0,1 mm bei **4/8**; Neigung ≲ 5 mrad; *jeder Fehlschlag wird begründet* |
| Unabhängiger Vergleich | pyFAI im Kaltstart scheitert an **11/11** derselben Bilder; von unserer Lösung gestartet stimmt es uns zu — der Gewinn liegt in der Initialisierung, nicht in einer anderen Engine |

<div align="center">
<img src="assets/fig2_frames_rings.png" alt="Benchmark-Bilder mit angepassten Ringen" width="760"><br>
<sub>Benchmark-Bilder mit angepassten Ringen: (a) CeO₂ bei 300 mm (8 zugeordnete Ringe); (b) Mittel der 52 statischen Al₂O₃-Bilder.</sub>
</div>

<details>
<summary><b>Weitere Belege</b> (CeO₂-Abstandsfehler · unabhängiger pyFAI-Vergleich)</summary>

<div align="center">
<img src="assets/fig3_ceria_errors.png" width="520"><br>
<sub>CeO₂-Fehler über sieben Abstände (native 2D-Verfeinerung): mittlerer und schlimmster |ΔD| = 0,1 mm.</sub><br><br>
<img src="assets/fig8_duel_ablation.png" width="760"><br>
<sub>Unabhängiger Vergleich mit pyFAI (Kaltstart P0 vs. diese Pipeline T, gleicher Start P1) und Komponenten-Ablation.</sub>
</div>

</details>

## Schnellstart

```bash
pip install numpy scipy matplotlib pyFAI fabio pillow h5py torch
```

```python
import fabio
from probes.solver_core import solve_image

img = fabio.open("ihr_bild.edf").data.astype(float)
img[img >= 1e9] = 0.0            # Sentinel-/Negativmaskierung (zoo-Schema)
img[img < 0] = 0.0

res = solve_image(img, px_m=172e-6)      # lam=None → gemeinsame λ–D-Lösung
print(res["phase"], res["sol"])          # Phasen-ID + Geometrielösung
```

Laden Sie die im Paper referenzierten öffentlichen Testbilder
(<https://www.silx.org/pub/pyFAI/testimages/>) nach `data/real/` und führen
Sie die Proben aus — jede Tabelle in `probes/results/` wird deterministisch
reproduziert (feste Seeds).

| Einstiegspunkt | Zweck |
|---|---|
| `probes/solver_core.py` | einheitlicher `solve_image()`-Einstieg (Canvas + Zero-Shot-sicherer Fallback) |
| `probes/zoo_run.py` | geräteübergreifender Benchmark (silx-Testbilder) |
| `probes/phase_map.py` | Bedingungskarte (Belichtung / Neigung / Spannweite) |
| `probes/ablation8.py` | Komponenten-Ablation (Stresssatz) |
| `probes/_pyfai_duel.py` | unabhängiger Vergleich mit pyFAI |
| `generators/train.py` | Training des NN-Initialisierers (synthetische Bilder) |

## Repository-Struktur

```
torchfai/      Solver-Bibliothek (Phasen, Simulator, Verfeinerung, Modell)
probes/        Benchmark-Proben, Abbildungsskripte, GUI-Solver
  results/     vollständige Ergebnistabellen (.md/.json/.txt) + Referenz-PONIs
generators/    Datengeneratoren für das NN-Training + Merge-Werkzeuge
docs/          technische Belegdokumente (Experimentprotokolle)
manuscript/    Manuskriptentwurf v0.5 (Markdown)
assets/        in diesem README verwendete Abbildungen
```

## Einladung zum Blindtest

> Wir sammeln **1–2 historische Bilder + zugehörige Referenz-.poni** von
> beliebigen Messplätzen — Detektor, Kalibrierphase, Abstand, Wellenlänge
> und Baujahr sind frei. Sie senden Bild und bekannte Kalibrierung; wir
> rechnen vollständig blind und liefern ΔD, Strahlzentrum, Neigungen, ein
> einsatzbereites PONI, Unsicherheitsintervalle und die Phasenidentifikation.
> Die Referenz sehen wir erst, nachdem unsere Lösung fixiert und
> zeitstempelbelegt ist. **Negative Ergebnisse werden genauso berichtet** —
> mit Begründung statt Ausrede.

## Zitation

```bibtex
@software{pyfai_zero_shot_2026,
  author = {{The pyfai-zero-shot authors}},
  title  = {pyfai-zero-shot: phase-blind, zero-shot geometry calibration
            of 2-D X-ray detectors from powder rings},
  year   = {2026},
  url    = {https://github.com/ziaoliu902-cyber/pyfai-zero-shot},
  note   = {v0.5; Zenodo DOI to be assigned}
}
```

## Lizenz und Danksagung

MIT (siehe [LICENSE](LICENSE)). Die Benchmark-Bilder sind die öffentlich
verteilten pyFAI-Testbilder (silx.org, aufgenommen an D2AM/BM02, ESRF);
dieses Projekt ist ein unabhängiger Nutzer von pyFAI und steht in keiner
Beziehung zum pyFAI-Team.

<sub>Das Manuskript (v0.5) ist ein Entwurf in Arbeit und wurde nicht
begutachtet. Die mehrsprachigen READMEs werden manuell gepflegt;
maßgeblich ist die englische Version.</sub>
