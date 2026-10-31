# Provenance

What is verified, what is missing, and what cannot be reproduced. Every claim
below names the evidence that establishes it.

---

## 1. The claim

The source in this repository is the **exact historical source** that produced
the 1,800 out-of-fold predictions deposited at DOI
[10.5281/zenodo.22173120](https://doi.org/10.5281/zenodo.22173120). It is not a
reimplementation, not a reconstruction, and not a cleaned-up equivalent.

Four independent lines of evidence:

### 1.1 The recorded commit matches the checkout

`data/metadata/phase6b/phase6b_summary.json` records the commit that produced
the deposited predictions:

```json
"provenance": { "git_commit": "ac39cc31b969f73624e9ef19cd1834a935db1f35", ... }
```

`data/metadata/dataset_summary.json` records the same commit for the dataset
build. `git rev-parse HEAD` in the parent repository returns exactly that
commit, and `git status` reports **no modification** to any tracked pipeline
module: `src/config.py`, `src/data/`, `src/preprocessing/`, `src/features/`,
`src/io/`, `src/eval/`, `src/qc/` and the tracked CLIs are byte-identical to
that commit.

### 1.2 Every copied file is byte-identical to its origin, with two stated exceptions

`MANIFEST.sha256` records the SHA-256 of all 104 files, and `MANIFEST.json`
classifies each one:

| Class | Count | Meaning |
|---|---|---|
| `verbatim` | **95** | byte-identical copy of the same path in the parent repository at commit `ac39cc31…`. Verified by hashing source and destination: **95 of 95 agree.** |
| `derived` | **2** | a copy whose machine-local absolute paths were made repository-relative before publication. Described below. |
| `authored` | **7** | written for this repository: `README.md`, `PROVENANCE.md`, `requirements.txt`, `LICENSE`, `.gitignore` and the two scripts under `scripts/`. |

No source module is `derived` or `authored`: all 50 pipeline modules, all 11
bridge modules and all 19 test files are `verbatim`.

**The two derived files, and exactly what changed in them.** Neither appears in
the Zenodo deposit, so publishing them is new, and both carried this machine's
absolute filesystem paths:

| File | Change | Verified unchanged |
|---|---|---|
| `data/metadata/recordings.csv` | the `path` column rewritten from, e.g., `J:\…\data\raw\S001\S001R04.edf` to `data/raw/S001/S001R04.edf`, in all 120 rows | every other column, including all 120 SHA-256 digests, byte sizes, sampling rates, durations and annotation counts — compared column by column before and after |
| `data/metadata/dataset_summary.json` | `config.paths.root` rewritten from an absolute local path to `"."` | every other field — compared as parsed JSON with that one key normalised |

The rewrite is cosmetic for a reproducer, whose paths will differ anyway, and it
removes a local filesystem layout from a public deposit. It is recorded here and
in `MANIFEST.json` rather than glossed over, because the byte-identity claim in
§1.1–§1.4 would otherwise be inaccurate for these two files.

**Line endings are not converted, and this matters.** `MANIFEST.sha256`
guarantees that a clone is bit-for-bit what was deposited, so any end-of-line
conversion on commit or checkout would break it on some platform. The parent
repository runs with `core.autocrlf=true` and no `.gitattributes`, so its Windows
working tree holds a mix: **30 of these text files are CRLF on disk and 71 are
LF**. `.gitattributes` here sets `* -text`, which disables conversion in both
directions, so each file is preserved exactly as it was. The mixed endings are a
property of the source faithfully carried over, not an oversight.

This was found by testing rather than assumed. An earlier `.gitattributes` using
`* text=auto eol=lf` normalised the 30 CRLF files on commit, and **39 of the 105
digests then failed to verify from a fresh clone** — under both a default clone
and one with `core.autocrlf=true`. The `-text` policy was adopted in response,
and both clone modes now verify 105 of 105.

### 1.3 The pipeline self-verifies against the published results

`src/encoder/paper1.py::_verify_reproduction` compares the per-subject balanced
accuracies it reproduces against the published
`data/metadata/phase4b/phase4b_subject_results.csv` and **raises** on any
disagreement above 1e-9. It passes: **35 of 40 subjects reproduce
bit-identically** and the worst disagreement is **1.11e-16**.

### 1.4 Re-execution reproduces the deposited predictions exactly

Measured by `scripts/reproduce_predictions.py`:

| Quantity | Result |
|---|---|
| Rows | 1,800 re-executed vs 1,800 deposited |
| `epoch_uid`, `subject_id`, `run_id`, `true_label` | **identical, in the same order** |
| Decoded side (the prediction) | **identical on all 1,800 — 0 differ** |
| `left_probability`, `right_probability` | max abs difference **1.11e-16** |
| mu-band `lateralisation` | max abs difference **4.44e-16** |
| Confusion matrix | **[[535, 370], [351, 544]]** on re-execution, deposit and manuscript |
| Accuracy | **0.599444** → 0.5994 ✓ |
| Balanced accuracy | **0.599491** → 0.5995 ✓ |
| Macro F1 | **0.599434** → 0.5994 ✓ |
| Subject-independence audit, 40 folds | no leakage detected |
| Dataset fingerprint | `80fc4eec2efd5206` |

---

## 2. Configuration verified against the manuscript

Each row was read from the source at run time by
`scripts/reproduce_predictions.py`, not transcribed from documentation.

| Manuscript statement | Source | Verified |
|---|---|---|
| 40-subject leave-one-subject-out | `src/eval/protocol.py::losocv_folds`; 40 folds constructed, 1,755 train / 45 test epochs each | ✓ |
| 21-channel sensorimotor ROI | `src/data/physionet_spec.py::SENSORIMOTOR_CHANNELS`; names and order confirmed | ✓ |
| 8–30 Hz filtering | `src/config.py` `l_freq=8.0`, `h_freq=30.0`, FIR, zero-phase | ✓ |
| `[0, 4)` s epoch | `epoch_tmin=0.0`, `epoch_tmax=4.0`, `include_tmax=False` → 640 samples at 160 Hz | ✓ |
| Logistic regression, L2, C=1, lbfgs, max_iter=5000 | `src/eval/baselines.py::logistic_regression` | ✓ |
| 1,800 binary epochs, 45 per subject | re-execution returns 1,800 posteriors over 40 subjects | ✓ |
| Common-average referencing | `common_average_reference=True` | ✓ |
| 96 features | `FoldProvenance.n_features = 96`, matching the recorded `provenance.paper1.n_features` | ✓ |
| Uncalibrated, ECE 0.169 | `audit.all_uncalibrated = True`; `provenance.paper1.uncalibrated_ece = 0.16933…` | ✓ |

One wording note: `LogisticRegression.penalty` reads as scikit-learn's
deprecation sentinel rather than the string `"l2"`, because `baselines.py`
deliberately does not pass `penalty` — naming it explicitly is deprecated from
scikit-learn 1.8, and the default is L2. The effective penalty is L2.

---

## 3. What is missing

### 3.1 The raw recordings, by design

`data/raw/**/*.edf` is **not** in this repository. EEGMMIDB is public, but
redistribution is not ours to grant: users must obtain it from PhysioNet. What
*is* included so a download can be verified:

* `data/SHA256SUMS.txt` — the official PhysioNet manifest, cached.
* `data/metadata/recordings.csv` — per-file SHA-256, byte size, sampling rate,
  channel count, duration, annotation count and acquisition date for all 120
  recordings (40 subjects × 3 runs).

The original build verified all 120 EDFs and their 120 `.edf.event` sidecars
against the official manifest.

### 3.2 The processed archives, by design

`data/processed/S0NN.npz` is **not** included: the 40 archives are large and are
fully rebuildable from the recordings by `python -m src.cli.build_dataset`.

### 3.3 No fitted model was ever persisted

There is no `.pkl`, `.joblib` or stored estimator anywhere in the original
project, because `run_losocv` fits inside each fold and lets the estimator go
out of scope. This is why the pipeline **re-executes** rather than loads. It is
a property of the original design, not a loss.

### 3.4 Thirty-five files were untracked in the parent repository

Several modules the decoder depends on were untracked in the parent repository
and therefore exist in **no git history at all**: all four modules of
`src/encoder/` — including `paper1.py`, which performs the re-execution —
together with `src/cli/phase6b_encoder.py`, `src/cli/phase4b_final.py`,
`src/cli/phase5_audit.py`, `src/cli/fetch_direct.py` and
`src/features/diagnostics.py`. Counting tests and artifacts, **35 of the 104
files** here had untracked origins; `MANIFEST.json` flags each one with
`origin_tracked_in_git: false`.

Publishing this repository is the first time those files enter version control.
That is the main preservation value of this deposit: until now the code that
produced the deposited predictions existed only on one machine.

---

## 4. What cannot be reproduced

| | |
|---|---|
| **Bit-identical posteriors** | L-BFGS can land on a different final bit between runs, library versions and platforms. The agreement is 1.11e-16 and the *predictions* are unaffected — a 1e-16 shift never crosses the 0.5 boundary. The pipeline's own check enforces 1e-9 against the published per-subject accuracies and aborts if exceeded. |
| **The raw → processed step, without the recordings** | Workflow B requires the PhysioNet download. Workflow A does not, which is why the out-of-fold predictions are included. |
| **Exact library versions** | The original run recorded Python 3.14.6, numpy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0 (decoder) and scipy 1.18.0, mne 1.12.1 (dataset build). `requirements.txt` pins these. Older or newer versions may shift the last bits of the posteriors further; the self-check is the guard. |
| **Precision, recall, F1 and the confusion matrix as *recorded* values** | The original run never stored them. They are recomputed from the stored predictions. The quantities it *did* store — per-subject balanced accuracy, posterior margin, cue agreement, usable-epoch fraction — reproduce within 2.3e-16, which is the warrant for trusting the recomputation. This distinction is stated wherever those four numbers appear. |
| **A commit identifying the archived Zenodo release** | The Zenodo record links no repository, tag or commit. The archive is identified by `md5:1539458a496176fcd98f70c0766effdb`, `sha256:7e233ca554ccd7aa475bb709c6403481b77ddb75aa29636a7b7e6fafbbeb69d4` and a 277-entry manifest, all of which verify against the local copy of the deposit. |

---

## 5. Relationship to the archived deposit

The archived deposit at DOI 10.5281/zenodo.22173120 contains the closed-loop
architecture, the derived neuro states and the 1,800 out-of-fold predictions,
**but not** the upstream preprocessing, feature-extraction or decoder source —
verified by searching its `MANIFEST.json`, which lists **zero** EEG-pipeline
modules among its 277 files. That gap is what this repository closes.

The deposit is unchanged by this work: **no new DOI was minted and no Zenodo
record was modified.** Section 6 of the repository's cleanup report lists the
files a future Zenodo version would need to include.

---

## 6. Reproducing this provenance check

```bash
python scripts/metrics_from_deposited.py      # workflow A, no EEG needed
python scripts/reproduce_predictions.py       # workflow B3, needs data/processed
python -m pytest -q                           # offline test suite
sha256sum -c MANIFEST.sha256                  # every file against its recorded digest
```

The digest check is expected to report **105 of 105 OK** immediately after a
clone, on Windows as well as on Linux or macOS. If it does not, the checkout has
altered the files and nothing else in this record should be trusted.

Measured on this repository: **417 tests passed, 1 skipped, 25 deselected.** The
single skip is the live-data suite, which reports `PHYSIONET DATA UNAVAILABLE`
and states in its own message that it is a skip and not a pass — the correct
outcome when the restricted recordings are absent. The 25 deselected are the
`physionet`-marked real-data tests, deselected by default in `pytest.ini`.
