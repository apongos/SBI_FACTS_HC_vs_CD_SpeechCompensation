# SBI + FACTS: Healthy Controls vs. Cerebellar Degeneration Speech Compensation

This repository uses **Simulation-Based Inference (SBI)** to fit the **Feedback Aware Control of
Tasks in Speech (FACTS)** model of speech motor control to auditory-perturbation behavioral data,
and to compare the inferred control parameters between **Healthy Controls (HC)** and speakers with
**Cerebellar Degeneration / ataxia (CD)**.

The FACTS simulator is a fork/adaptation of [`kwangsk/FACTS`](https://github.com/kwangsk/FACTS)
(Parrell et al., 2019; Kim et al 2023.). On top of the simulator, we:

1. Define a FACTS "simulator" that maps 9 free control parameters (`theta`) to a simulated F1
   compensation trajectory (`x`).
2. Train an amortized neural posterior estimator (`sbi`, NPE/SNPE) on ~1M simulations.
3. Optionally fine-tune group-specific posteriors with multi-round inference.
4. Sample the posterior conditioned on the empirical HC and CD compensation curves
   (Parrell et al., 2017) and compare parameters between groups.

Everything is driven from the Jupyter notebooks in the repo root.

---

## 1. Requirements

- **Conda / Miniforge / Mambaforge** (the environment is defined as a conda `environment.yml`).
- macOS (Apple Silicon), Linux, or Windows. The Maeda articulatory synthesizer ships as
  precompiled extensions in `FACTS_Modules/` (e.g. `_maeda.cpython-311-darwin.so` for macOS
  Python 3.11, plus `.so`/`.pyd` builds for other platforms/versions). If none matches your
  platform you may need to rebuild them (see [Troubleshooting](#7-troubleshooting)).
- ~2 GB free disk (the repo already includes pretrained posteriors and simulation shards).

The environment pins **Python 3.11** and key scientific/ML packages including
`sbi=0.25.0`, `pytorch=2.5.1`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`,
`scikit-learn`, and `arviz`.

---

## 2. Setup

From the repository root:

```bash
# Create the environment (named "sbi311")
conda env create -f environment_FACTS_SBI_py311.yml

# Activate it (do this in every new terminal session)
conda activate sbi311
```

> If `conda env create` is slow or fails to solve, try [mamba](https://mamba.readthedocs.io/):
> `mamba env create -f environment_FACTS_SBI_py311.yml`.

Register the environment as a Jupyter kernel (optional but recommended) and launch Jupyter:

```bash
python -m ipykernel install --user --name sbi311 --display-name "Python (sbi311)"
jupyter notebook   # or: jupyter lab
```

Then open one of the notebooks and select the **Python (sbi311)** kernel.

---

## 3. Which notebook to run

- **`SBI_FACTS_FINAL_FOR_PAPER_FEB2_2026_Load_Train_Plot-python311-Updated_with_MultiRound.ipynb`**
  — **Recommended / most up to date.** Amortized training plus multi-round, group-specific
  fine-tuning (separate HC and CD posteriors).
- `SBI_FACTS_FINAL_FOR_PAPER_FEB2_2026_Load_Train_Plot-python311.ipynb`
  — Earlier version without the multi-round fine-tuning step.

Both notebooks must be run **from the repository root** so that relative paths (config `.ini`,
`GesturalScores/`, `sbi_resources/`, `FACTS_Modules/`) resolve correctly.

---

## 4. Notebook workflow (what each stage does)

Run the cells top to bottom. The main stages are:

1. **Imports & seeds** — imports FACTS modules and `sbi`, sets a fixed seed (42) for reproducibility.
2. **Define the `FACTS(theta)` simulator** — reads the base config
   `DesignC_AUKF_onlinepertdelay_SBI_Al.ini` and the gestural score
   `GesturalScores/KimetalOnlinepert2.G`, overrides the 9 free parameters from `theta`, runs a
   FACTS simulation, and returns the simulated F1 compensation trajectory.
3. **Parameter labels & priors** — the 9 inferred parameters and their uniform prior bounds
   (see [Parameters](#5-inferred-parameters)).
4. **Load training simulations** — `load_sharded_npz('sbi_resources/all_sbi_simulations_v1/')`
   concatenates the `simulations_part*.npz` shards into `all_theta` (Nx9) and `all_x_ds` (Nx140).
5. **Load empirical data** — `sbi_resources/parrell_2017/parrell_data.mat` (Parrell et al., 2017),
   resampled to 5 ms frames and aligned to the simulator output (HC and CD mean curves + CIs).
6. **Quality checks / Figure 1** — plots the closest training trajectories to the HC and CD data.
7. **Train the neural posterior** — trains an NPE/SNPE density estimator (`learning_rate = 5e-6`,
   batch size 128). **This can take 12+ hours (~180–250 epochs).** You can skip training.
8. **(Optional) Multi-round fine-tuning** — fine-tunes separate HC and CD posteriors.
9. **Save / load posteriors** — controlled by `SAVE`/`LOAD` flags. By default the notebook is set
   to **`LOAD = True`, `SAVE = False`**, so it loads the pretrained posteriors shipped in
   `sbi_resources/` instead of retraining.
10. **Sample & analyze** — draws posterior samples for HC and CD, finds posterior modes,
    re-simulates FACTS at those modes, computes RMSE against the empirical curves, and produces the
    comparison plots/tables.

### Fastest path (skip the long training)

To reproduce the analysis/figures without the multi-hour training, run every cell **except** the
training cells, making sure the save/load cell keeps:

```python
SAVE = False
LOAD = True
```

This loads the pretrained posteriors (`sbi_resources/Test_multiroundposterior{HC,CD}_*.pkl`) and
proceeds straight to sampling and plotting.

---

## 5. Inferred parameters

The 9 free parameters (`theta`) and their uniform prior bounds:

| # | Parameter | Prior min | Prior max |
|---|-----------|-----------|-----------|
| 1 | Auditory noise scale (estimator) | 1e-8 | 20 |
| 2 | Somatosensory noise scale (estimator) | 1e-8 | 20 |
| 3 | Task state estimator (TSE) process scale | 1e-8 | 0.05 |
| 4 | Articulatory state estimator (ASE) process scale | 1e-8 | 1e-4 |
| 5 | Auditory delay (ms) | 100 | 240 |
| 6 | Somatosensory delay (ms) | 100 | 240 |
| 7 | Natural frequency | 3 | 10 |
| 8 | Artic-to-task noise scale | 0 | 0.20 |
| 9 | ArticSFCLaw noise scale | 0 | 1000 |

Each `theta` is written into the corresponding fields of the `.ini` config before a simulation.

---

## 6. Repository layout

```
.
├── environment_FACTS_SBI_py311.yml          # Conda environment (name: sbi311)
├── DesignC_AUKF_onlinepertdelay_SBI_Al.ini  # Base FACTS config used by the SBI simulator
├── global_variables.py                      # FACTS constants (dimensions, neutral attractors)
├── facts_visualizations.py                  # Plotting helpers for single/multi-trial results
├── SBI_FACTS_..._MultiRound.ipynb           # Main notebook (recommended)
├── SBI_FACTS_..._python311.ipynb            # Earlier notebook (no multi-round)
├── FACTS_Modules/                           # FACTS simulator (model, estimators, Maeda synth, LWPR/DNN models)
├── GesturalScores/                          # Gestural scores (.G), e.g. KimetalOnlinepert2.G
├── Maedadata/                               # Maeda synthesizer support data (area/sig/specdata)
└── sbi_resources/
    ├── all_sbi_simulations_v1/              # Sharded training simulations (theta/x) + manifest/metadata
    ├── parrell_2017/                        # Empirical HC/CD data (parrell_data.mat, magnitudes.mat)
    └── Test_*posterior*.pkl / *inference*.pkl  # Pretrained SBI posteriors/inference objects
```

---

## 7. Troubleshooting

- **`ModuleNotFoundError` for FACTS modules** — make sure you launched Jupyter from the repo root
  and selected the `sbi311` kernel.
- **`ImportError` loading `_maeda` / `_relokate`** — the compiled extension for your platform/Python
  version may be missing. Precompiled builds live in `FACTS_Modules/`; if yours isn't present you'll
  need to rebuild them (see the SWIG sources `relokate.i`, `relokatesetup.py`, `numpy.i`) or run the
  matching Python version (this env targets Python 3.11).
- **`FileNotFoundError` for `sbi_resources/...`** — confirm the notebook working directory is the
  repo root and that the `sbi_resources/` shards/pickles are present.
- **`sbi` API differences** — the notebooks target `sbi=0.25.0` (NPE is the current name for SNPE);
  the pinned environment avoids version-mismatch errors.
- **Stray `pdb.set_trace()`** — the simulator (`FACTS_Modules/Model.py`) contains debug breakpoints
  in some code paths. If a run pauses at a `(Pdb)` prompt, type `c` to continue, or comment out the
  breakpoint.

---

## 8. References

- Parrell, B., Ramanarayanan, V., Nagarajan, S., & Houde, J. (2019). *The FACTS model of speech
  motor control.* PLOS Computational Biology.
- Parrell, B., et al. (2017). Auditory perturbation study providing the HC/CD compensation data used
  here (`sbi_resources/parrell_2017`).
- Kim, K. S., et al. FACTS Design C / AUKF extensions — see [`kwangsk/FACTS`](https://github.com/kwangsk/FACTS).
- Tejero-Cantero, A., et al. (2020). *sbi: A toolkit for simulation-based inference.* JOSS.
