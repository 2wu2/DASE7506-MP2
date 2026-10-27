# DASE7506 MP2 - Version 4 submission

**Author:** Zhu Jinting  
**Student ID:** 3036709066  
**Official test accuracy:** `84.36830835117773` percent (`394 / 467`)  
**Method:** `code/evidence_method.py`  
**Checkpoint:** `code/runs/evidence_v4/model_bundle/`

This package contains the final v4 evidence-fusion method, its frozen assets, a six-page English report, complete supplied data, and experiment records. It has not been uploaded or submitted automatically.

## Contents

- `REPORT.pdf`: six-page English coursework report.
- `REPORT.md`: editable English report text.
- `report/build_report.py`: optional PDF authoring source; it does not participate in inference.
- `code/`: frozen inference/training source and official evaluation tools.
- `code/data/`: complete supplied training, validation and test banks, with original license and attribution.
- `code/runs/evidence_v4/model_bundle/`: all inference assets (24,191 bytes uncompressed).
- `code/runs/evidence_v4/test-score.json`: official final score.
- `code/runs/evidence_v4/frozen.json`: hashes saved before the v4 test.
- `code/runs/evidence_v4/`: protocols, folds, comparisons and other v4 records. Training-analysis arrays are not used at inference.
- `records/history/`: earlier versions' metrics and development records.
- `records/package-reproduction.json`: packaging-time reproduction checks.
- `provenance/`: supplied assignment instructions, original requirements and original release manifest.
- `RELEASE.json`, `SHA256SUMS.json`: exact version metadata and delivered-file checksums.

## Install (Python 3.12)

The runtime has one third-party dependency: NumPy. The tested version is pinned to `numpy==2.3.5` in `code/requirements.txt`. The supplied course pin was `numpy==2.5.3`; it is retained in `provenance/ORIGINAL_REQUIREMENTS.txt`. The score was measured with Python 3.12.14 / NumPy 2.3.5 on Windows 11. The course pin has not been independently verified.

From the extracted package root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r code/requirements.txt
cd code
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
python check_data.py
python evaluate_evidence.py
```

On Linux/macOS:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r code/requirements.txt
cd code
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python check_data.py
python evaluate_evidence.py
```

After installing the dependency, prediction and scoring run offline. No API key, pretrained weights or retraining are required.

Expected official output:

```json
{"correct": 394, "n": 467, "accuracy": 84.36830835117773}
```

The wrapper first checks the pre-test hashes and launches a fresh prediction process plus an independent scoring process. It writes `runs/evidence_v4/test-predictions.json`, `test-score.json` and `test-timing.json`. The original recorded full-process time is 2.809 seconds on an Intel Core Ultra 9 290HX Plus. The reference Xeon Platinum 8457C has not been verified.

Equivalent official commands, with thread variables already set:

```bash
python predict.py --method evidence_method.py --checkpoint runs/evidence_v4/model_bundle --split test --output runs/reproduced-test-predictions.json
python score_predictions.py --split test --predictions runs/reproduced-test-predictions.json --output runs/reproduced-test-score.json
```

## Reproduce training

Use a separate extracted copy for retraining because generated run files are overwritten. From `code/`, with the same environment and thread settings:

```bash
python check_evidence.py
python audit_evidence.py
python train_evidence.py
python verify_evidence.py
```

The audit step creates a training feature cache and its source hash. Training uses the supplied training split with three outer folds, three inner folds for fusion, and seed `20261027`. It freezes the selected model before evaluating validation. It does not load test data or test scores. `verify_evidence.py` checks the frozen hashes, candidate permutations, new IDs, batch independence, asset-only inference and official validation scoring. Its optional original-source comparison can be enabled by setting `MP2_ORIGINAL_CODE_DIR`; this is not needed for prediction.

The trained model is three small linear scorers (base, evidence and fusion), not an LLM. A fresh final training run is approximately 40 seconds plus 13 seconds for feature extraction on the recorded machine; timings vary by machine. Detailed hyperparameters and all seven comparisons are in `code/runs/evidence_v4/protocol.json` and `cv-results.json`.

## Integrity and version identity

From the package root, run:

```bash
python verify_package.py
```

This checks every file listed in `SHA256SUMS.json`. The frozen inference, feature, training source and checkpoint are byte-identical to the v4 pre-test files. The evaluation/verification wrappers normalize Windows-style separators in the saved hash manifest, and the optional original-directory lookup uses an environment variable. These packaging changes are outside the frozen method and do not affect predictions. The dependency pin and documentation describe the environment actually tested.

Run the package-integrity check before training or evaluation commands that overwrite records. A later manifest mismatch in generated JSON may simply reflect a new timestamp or timing; the frozen code and model hashes must still match. The original release manifest in `provenance/` describes the instructor's original package, not this submission archive.

## Inference asset archive

The separate checkpoint ZIP contains `model_bundle/base.npz`, `rules.npz`, `meta.npz` and `config.json`, plus its checksum list and an English usage note. The sum of the four inference assets is 24,191 bytes. If using that archive, extract `model_bundle/` into `code/runs/evidence_v4/` and use the matching `evidence_method.py` from this release.

## Development history and disclosure

All earlier test outcomes are retained. Development continued after earlier aggregate test scores were known. Later fitting used training data, and model selection used validation or training out-of-fold criteria rather than per-question test labels. This is not a pristine one-shot held-out-test history; the report states this limitation explicitly. No answer lookup, ID-based rule or cached prediction is used by the method.

OpenAI Codex provided substantive assistance with implementation, feature design, experiments, code review, analysis, report drafting and packaging. The official course input/score tools and GSM8K-derived data are reused and attributed. The student is responsible for understanding and verifying the submission. See `REPORT.pdf`, `provenance/`, and `code/data/LICENSE.txt` / `NOTICE.txt`.

## Submit

Use student ID `3036709066` and score `84.36830835117773`. Publish this exact code version, English report and records in a publicly accessible repository, and provide the matching checkpoint link. Follow the course website's GitHub issue creation step to save the submission. This local ZIP alone is not an online submission.
