# Upstream EEG decoder — motor-imagery decoding pipeline

The upstream motor-imagery EEG decoding pipeline for

> **Evidence-Bounded Decision and Actuation for Closed-Loop Adaptive Virtual
> Reality**

and the code that produced the 1,800 out-of-fold predictions deposited in the
accompanying research archive, **DOI
[10.5281/zenodo.22173120](https://doi.org/10.5281/zenodo.22173120)**.

This is the *historical* source, not a reimplementation. Every file here is a
byte-identical copy of the code that generated the deposited predictions, and
every copy's SHA-256 is recorded in `MANIFEST.sha256`. See
[`PROVENANCE.md`](PROVENANCE.md) for what that claim rests on and what it does
not cover.

---

## What this decoder is, and what it is not

It decodes **left- versus right-hand motor imagery** from the public
[PhysioNet EEG Motor Movement/Imagery Database](https://physionet.org/content/eegmmidb/1.0.0/)
(EEGMMIDB v1.0.0), subjects S001–S040, runs R04/R08/R12, under
leave-one-subject-out cross-validation.

Its measured performance is modest and heterogeneous, and that is the point:

| Metric | Value |
|---|---|
| Accuracy (pooled over 1,800 held-out epochs) | **0.5994** |
| Balanced accuracy | **0.5995** |
| Macro F1 | **0.5994** |
| Confusion matrix (rows true L/R, cols predicted L/R) | **[[535, 370], [351, 544]]** |
| Per-subject balanced accuracy | mean **0.5988**, SD 0.1218, range **0.398–0.932** |
| Subjects at or above chance | **32 of 40** |
| Calibration | **none**; recorded uncalibrated ECE **0.169** |

The architecture that consumes this decoder was designed around those numbers
rather than in spite of them: the decoder's output enters as an explicitly typed
evidence channel whose unreliability is represented rather than hidden. The
decoder is **not** decision-critical in the architecture's evaluated
configuration. If you are looking for a competitive motor-imagery decoder, this
is not one, and no claim is made that it is.

**No clinical, physiological or diagnostic claim is made or supported.**

---

## Installation

Python 3.11 or newer. The pipeline was developed and validated on the versions
pinned in `requirements.txt`.

```bash
git clone https://github.com/Quddos/upstream-EEG-decoder.git
cd upstream-EEG-decoder
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

On Windows, prefer `python -m pytest` over a bare `pytest`: the Scripts
directory is often not on `PATH`.

---

## Two reproduction workflows

### A. Reproduce the reported metrics from the deposited predictions

**No EEG recordings needed.** The deposited out-of-fold predictions are included
under `data/metadata/phase6b/`, so the reported metrics can be recomputed
immediately:

```bash
python scripts/metrics_from_deposited.py
```

This reads only `phase6b_neurostates.csv` (1,800 rows, one per held-out epoch,
carrying the true label and the leave-one-subject-out decoded side), recomputes
accuracy, balanced accuracy, per-class and macro precision/recall/F1 and the
confusion matrix, and cross-checks the per-subject quantities the original run
*did* store. Runs in about a second.

Note on what is retrieved and what is recomputed: the original run stored
per-subject balanced accuracy, posterior margin, cue agreement and usable-epoch
fraction, but **not** precision, recall, F1 or the confusion matrix. Those four
are therefore recomputed from the stored predictions. The script prints the
agreement on the stored quantities — all within 2.3e-16 — which is what licenses
trusting the rest.

### B. Re-execute the decoder end to end

**Needs the EEG recordings.** They are not redistributed here: PhysioNet's terms
require users to obtain the data from the original source.

**B1 — obtain the recordings.** Download EEGMMIDB v1.0.0 subjects S001–S040,
runs R04, R08 and R12 from
<https://physionet.org/content/eegmmidb/1.0.0/> into `data/raw/S001/…`:

```bash
python -m src.cli.fetch_dataset --subjects all --runs imagery
```

`data/SHA256SUMS.txt` is the official PhysioNet manifest, cached here. Verify
what you downloaded against it:

```bash
python -m src.cli.verify_dataset --subjects all --fetch-events --fetch-manifest
```

**B2 — build the processed epoch archives.** Band-pass, re-reference, epoch:

```bash
python -m src.cli.build_dataset --subjects all --source physionet
```

This writes `data/processed/S001.npz … S040.npz` plus
`data/metadata/epoch_index.csv`, `dataset_summary.json` and `recordings.csv`.
The copies of those three metadata files shipped here are from the original
build; a fresh build should reproduce them.

**B3 — re-execute the decoder and compare against the deposit.**

```bash
python scripts/reproduce_predictions.py
```

It prints the configuration it is about to run (read from the source, not from
this README), re-executes the 40-fold leave-one-subject-out protocol, keeps the
per-epoch posterior the original run computed and discarded, compares all 1,800
rows against the deposited predictions, and recomputes the four metrics from
both. Expect roughly two to three minutes.

If your processed archives live elsewhere, pass the project root that contains
them:

```bash
python scripts/reproduce_predictions.py --data-root /path/to/project
```

**B4 — regenerate the deposited neuro-state artifacts.** The CLI that wrote
`phase6b_neurostates.csv` and the per-subject JSONL state files:

```bash
python -m src.cli.phase6b_encoder
```

---

## Validation already performed

Run on this source against the deposited artifacts:

| Check | Result |
|---|---|
| All 1,800 out-of-fold predictions reproduce | **exactly — 0 of 1,800 differ** |
| Epoch identity, subject, run and true label | **identical, in the same order** |
| Class posteriors | agree to **1.11e-16**; not bit-identical (see below) |
| mu-band lateralisation feature | agrees to **4.44e-16** |
| Confusion matrix | **[[535, 370], [351, 544]]** on the re-execution, the deposit and the manuscript |
| Accuracy / balanced accuracy / macro F1 | **0.599444 / 0.599491 / 0.599434** → 0.5994 / 0.5995 / 0.5994, matching the reported values |
| Self-verification against the published per-subject values | **passes**; 35 of 40 subjects bit-identical, worst disagreement **1.11e-16** |
| Subject-independence audit over 40 folds | **no leakage detected** |

The posteriors are not bit-identical because L-BFGS can land on a different
final bit between runs and platforms. The *predictions* are unaffected — a
1e-16 shift never crosses the 0.5 decision boundary — and the pipeline's own
self-check enforces a 1e-9 tolerance against the published per-subject
accuracies and aborts if it is exceeded.

### Test suite

```bash
python -m pytest -q                 # offline: synthetic fixtures, no network
python -m pytest -m physionet -v    # real-data tests; needs data/raw populated
```

Nineteen test files are included, covering the dataset specification,
preprocessing, feature extraction, the loader, quality control, the
leave-one-subject-out protocol, and — importantly — **leakage and circularity
controls** with deliberately leaky positive controls that must be caught.
Real-data tests are marked `physionet` and are deselected by default; when the
data is absent they **skip** rather than pass.

---

## Configuration, in full

Read from the source by `scripts/reproduce_predictions.py`, which prints it
before every run.

### Dataset and channels

| | |
|---|---|
| Dataset | PhysioNet EEGMMIDB v1.0.0 |
| Subjects | S001–S040 (40 of the 109 released) |
| Runs | R04, R08, R12 |
| Sampling rate | 160 Hz |
| Montage | 64 channels, 10–10 system |
| Epoch window | `[0, 4)` s from cue onset, end-exclusive → **640 samples** |
| Epochs, binary task | **1,800** (45 per subject) |
| Class balance | 905 LEFT / 895 RIGHT |

The epoch window starts at cue onset because any pre-onset baseline window would
fall inside the preceding trial and import its class.

### The 21-channel sensorimotor region of interest

`sensorimotor_21`, in this order:

```
FC5  FC3  FC1  FCz  FC2  FC4  FC6
C5   C3   C1   Cz   C2   C4   C6
CP5  CP3  CP1  CPz  CP2  CP4  CP6
```

Three rows — fronto-central, central, centro-parietal — each spanning left
hemisphere through the midline to the right. The restriction is applied **at
model-input time**, not at storage time: the `.npz` archives keep all 64
channels, so the 64-versus-21 comparison stays available. Two documented reasons
for the region: it cuts the input dimension threefold, which matters when a
subject contributes 45 binary epochs; and an upstream quality-control analysis
found every peak-to-peak amplitude threshold crossing in the examined run fell
*outside* it — frontopolar, occipital and frontal, never on an FC/C/CP electrode.

### Preprocessing

| Step | Setting |
|---|---|
| Band-pass | 8–30 Hz |
| Filter | FIR, **zero-phase** (therefore non-causal: this is an offline pipeline) |
| Transition bands | automatic |
| Notch | **off** — 8–30 Hz already places the power-line frequency deep in the stopband |
| Reference | **common average**, on |
| Epoching | `[0, 4)` s, `include_tmax=False` |
| Baseline correction | none |
| Artifact rejection | **none; no epoch is excluded.** A per-epoch peak-to-peak flag is recorded as advisory metadata and never used to reject |
| Feature-time normalisation | none; features are computed on un-normalised volts |
| Seed | 20260815 |

### Features — what the classifier actually receives

**96 spectral band-power features per epoch.** Not raw filtered samples, not
common spatial patterns, not a learned representation.

Welch periodogram, 256-sample segments (1.6 s, 0.625 Hz bins); bands mu `[8,13)`
Hz (8 bins) and beta `[13,30)` Hz (27 bins); passband `[8,30)` Hz as the
denominator for relative power.

| Family | Definition | Count |
|---|---|---|
| `{ch}_log_{band}` | log absolute band power | 42 |
| `{ch}_rel_{band}` | band power / passband power, weighted by **bin counts**, not nominal bandwidths | 42 |
| `lateralization_{band}` | log P(C4) − log P(C3) | 2 |
| `c3_minus_c4_log_{band}` | the reverse difference | 2 |
| `roi_mean_log_{band}` | mean log power over the 21 electrodes | 2 |
| `roi_std_log_{band}` | SD of log power across the 21 electrodes | 2 |
| `erd_C3_vs_roi_{band}`, `erd_C4_vs_roi_{band}` | hand-knob log power minus the region mean | 4 |
| **Total** | | **96** |

The last family is a **spatial proxy** for event-related desynchronisation, not
classical ERD, which would need the pre-cue baseline this epoching deliberately
does not retain. Two baseline substitutes were considered and rejected:
referencing to the held-out subject's own rest epochs uses that subject's labels,
and referencing to the recording mean makes a feature depend on which other
epochs are present. Every column is a function of a single epoch only.

### Classifier and protocol

```python
Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000,
                                 random_state=20260815)),
])
```

L2 penalty (the scikit-learn default; not passed explicitly, because naming it
is deprecated from scikit-learn 1.8). **No hyperparameter was selected from
data** — with this cohort size, selection inside a leave-one-subject-out loop
would be driven by three or four training subjects per inner fold, and selecting
on the outer fold would be tuning on test subjects. These are honest baselines,
not best-achievable numbers.

Protocol: **leave-one-subject-out, 40 folds**, 1,755 training and 45 test epochs
per fold. The standardiser sits *inside* the pipeline, so it is fitted on the
training subjects only; `tests/test_leakage.py` asserts this and includes a
deliberately leaky control that must be caught. Posteriors are **uncalibrated**
in all 40 folds.

### Re-execution, not retraining

The original study never persisted a fitted model: `run_losocv` fits inside each
fold, records fold summaries, and lets the estimator go out of scope. There is
no `.pkl`, `.joblib` or stored posterior anywhere. `src/encoder/paper1.py`
therefore **re-executes** the frozen modules with the frozen configuration,
features, folds and hyperparameters, and adds exactly one thing the original did
not: it calls `predict_proba` on the already-fitted pipeline and keeps the
result.

That this is re-execution and not retraining is **checked, not asserted**:
`_verify_reproduction` compares the reproduced per-subject balanced accuracies
against the published `phase4b_subject_results.csv` and raises if they disagree
beyond 1e-9.

---

## How this relates to the archive and the manuscript

```
PhysioNet EEGMMIDB (not redistributed)
        │   src/cli/fetch_dataset, verify_dataset
        ▼
data/raw/**.edf                       ← restricted; obtain from PhysioNet
        │   src/cli/build_dataset  →  src/preprocessing
        ▼
data/processed/S0NN.npz  +  data/metadata/epoch_index.csv
        │   src/features/{bandpower,extract}  →  96 features
        │   src/eval/{baselines,protocol}     →  40-fold LOSO
        │   src/encoder/paper1.encode_cohort  →  per-epoch posteriors
        ▼
data/metadata/phase6b/phase6b_neurostates.csv   ← 1,800 out-of-fold predictions
        │                                          DEPOSITED in the archive
        │   src/encoder/adapter, src/cli/phase6b_encoder
        ▼
neuro evidence channel  →  the closed-loop architecture (separate repository)
```

| Layer | Where it lives |
|---|---|
| This EEG pipeline | **this repository** |
| The 1,800 out-of-fold predictions and derived neuro states | deposited in the archive, and included here under `data/metadata/phase6b/` |
| The closed-loop architecture, decision engine, Unity client, ablations | the archive at DOI 10.5281/zenodo.22173120 |
| The manuscript | cites the archive; not part of either repository |

Eleven modules under `src/twin/` are included here because
`src/encoder/adapter.py` and `src/cli/phase6b_encoder.py` import them to write
the deposited neuro-state files. They are architecture code, already published
in the archive, duplicated here only so that this repository is self-contained
and `python -m src.cli.phase6b_encoder` runs. Neither the decoder nor either
reproduction script depends on them.

---

## Data provenance and what is deliberately absent

| | |
|---|---|
| Raw recordings (`data/raw/**/*.edf`) | **not included.** Public but restricted redistribution; obtain from PhysioNet |
| Processed archives (`data/processed/*.npz`) | **not included.** Derived from the recordings; rebuild with `build_dataset` |
| Out-of-fold predictions | **included** — this is what makes workflow A possible without the recordings |
| Per-file SHA-256 of every recording | **included** in `data/metadata/recordings.csv` |
| Official PhysioNet manifest | **included** as `data/SHA256SUMS.txt` |

The EEG recordings come from human participants in a public dataset. The motor
trajectories used by the downstream architecture are **simulated** and come from
a different source entirely; the two are separate evidence channels, were not
acquired together, are not from the same person, and no physiological coupling
between them is claimed or demonstrated.

---

## Citing

Cite the archive, which carries the DOI:

```
Qudus, R. and Pradhan, A. K. (2026). NeuroAdaptive Mirror World:
Evidence-Bounded Architecture for Closed-Loop Adaptive Virtual Reality —
Software and Reproducibility Artifacts. Zenodo.
https://doi.org/10.5281/zenodo.22173120
```

And the dataset:

```
Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N. and Wolpaw, J. R.
(2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.
IEEE Transactions on Biomedical Engineering 51(6), 1034–1043.

Goldberger, A., Amaral, L., Glass, L. et al. (2000). PhysioBank, PhysioToolkit,
and PhysioNet. Circulation 101(23), e215–e220.
```

## Licence

Code: MIT, see `LICENSE`. The EEG recordings are not included and remain
subject to the PhysioNet terms of use.
