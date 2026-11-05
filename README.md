# pyfai-zero-shot (v0.5)

Phase-blind, zero-shot geometry calibration of two-dimensional X-ray
detectors from a single powder-ring frame — no calibrant identity, no
initial geometry, and (in joint mode) no wavelength. Outputs a standard
pyFAI PONI plus bootstrap uncertainties.

## Layout

    torchfai/          solver library (phases, simulator, refine, model)
    probes/            benchmark probes, figure scripts, GUI solver
      results/         full result tables (.md/.json/.txt) and reference
                       PONIs backing every number in the paper
    generators/        NN-training data generators + merge tools
    docs/              technical evidence documents (experiment logs)
    manuscript/        paper draft (Markdown, v0.5)

## Key entry points

    probes/solver_core.py    unified solve_image() entry (canvas 520 +
                             zero-shot-safe fallback)
    probes/zoo_run.py        cross-instrument benchmark (silx test images)
    probes/phase_map.py      condition-map experiment (exposure/tilt/span)
    probes/ablation8.py      component ablation (stress set)
    probes/_pyfai_duel.py    independent comparison with pyFAI
    probes/fig_paper_render.py, fig1_arch.py, fig7_8_section46.py
                            all paper figures
    generators/train.py     NN initializer training (synthetic frames)

## Reproduce the benchmark

1. `pip install numpy scipy matplotlib pyFAI fabio pillow h5py torch`
2. Download the public test images referenced in the paper
   (https://www.silx.org/pub/pyFAI/testimages/) into `data/real/`.
3. Run the probes above; every table in `probes/results/` is
   regenerated deterministically (seeds fixed).

## Data

No data are bundled. The real frames are the publicly distributed pyFAI
test images (silx.org, D2AM/BM02 ESRF). Synthetic frames are generated
by `torchfai/simulator.py` + `generators/`.

## License

MIT (see LICENSE). The manuscript draft is provided for transparency.

## Citation

Repository: https://github.com/ziaoliu902-cyber/pyfai-zero-shot
[DOI placeholder — filled at deposition time]
