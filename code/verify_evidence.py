"""Frozen v4 inference checks and official validation, never test data."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '4'
import builtins
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np
from common import load_inputs, parse_answer
from evidence_method import predict


def main():
    root = Path('runs/evidence_v4'); bundle = root/'model_bundle'
    frozen = json.loads((root/'frozen.json').read_text())
    for name, digest in frozen.items():
        assert hashlib.sha256(Path(name.replace('\\', '/')).read_bytes()).hexdigest() == digest, name
    bank, arrays = load_inputs('validation')
    full = predict(bank, arrays, bundle)
    batch = []
    for start in range(0, len(bank), 37):
        batch.extend(predict(bank[start:start+37], {'embeddings': arrays['embeddings'][start:start+37]}, bundle))
    assert batch == full
    rng = np.random.default_rng(903)
    perm = np.array([rng.permutation(4) for _ in bank])
    pb = [dict(q, id=f'audit-new-id-{i}', candidates=[dict(q['candidates'][int(k)], index=j)
          for j, k in enumerate(perm[i])]) for i, q in enumerate(bank)]
    pa = {'embeddings': np.take_along_axis(arrays['embeddings'], perm[:, :, None], axis=1)}
    pp = predict(pb, pa, bundle)
    old = [parse_answer(q['candidates'][j]['text']) for q, j in zip(bank, full)]
    new = [parse_answer(q['candidates'][j]['text']) for q, j in zip(pb, pp)]
    assert old == new
    assert all(a is not None for a in old)
    assert all(type(j) is int and 0 <= j < 4 for j in full)
    assert predict([], {'embeddings': np.empty((0, 4, 1024))}, bundle) == []
    allowed = {str(p.resolve()) for p in bundle.iterdir()}
    original_open, original_io_open = builtins.open, io.open
    def guard(original):
        def call(file, *args, **kwargs):
            if not isinstance(file, int) and str(Path(file).resolve()) not in allowed:
                raise AssertionError(f'Unexpected inference file access: {file}')
            return original(file, *args, **kwargs)
        return call
    with patch('builtins.open', guard(original_open)), patch('io.open', guard(original_io_open)):
        assert predict(bank[:31], {'embeddings': arrays['embeddings'][:31]}, bundle) == full[:31]
    # Reference implementation files remain byte-for-byte equal to the
    # original supplied package; no scoring or numeric parsing changes.
    original = Path(os.environ.get('MP2_ORIGINAL_CODE_DIR', 'original-official-code'))
    scorer_unchanged = None
    if original.exists():
        scorer_unchanged = all(Path(n).read_bytes() == (original/n).read_bytes()
                               for n in ('common.py', 'predict.py', 'score_predictions.py'))
        assert scorer_unchanged
    tick = time.perf_counter()
    subprocess.run([sys.executable, 'predict.py', '--method', 'evidence_method.py',
                    '--checkpoint', str(bundle), '--split', 'validation',
                    '--output', str(root/'validation-predictions.json')], check=True)
    subprocess.run([sys.executable, 'score_predictions.py', '--split', 'validation',
                    '--predictions', str(root/'validation-predictions.json'),
                    '--output', str(root/'validation-score.json')], check=True)
    checks = dict(frozen_hashes=True, batch_independence=True, whole_bank_candidate_permutation=True,
                  new_ids=True, empty_bank=True, asset_only_inference=True,
                  official_scorer_unchanged=scorer_unchanged, all_choices_valid=True,
                  validation_fresh_process_seconds=time.perf_counter()-tick,
                  reference_cpu_verified=False, test_accessed=False)
    (root/'checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
