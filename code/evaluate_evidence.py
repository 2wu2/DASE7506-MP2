"""Official full test of frozen v4, including fresh process wall time."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    os.chdir(Path(__file__).resolve().parent)
    root = Path('runs/evidence_v4')
    frozen = json.loads((root/'frozen.json').read_text())
    for name, digest in frozen.items():
        if hashlib.sha256(Path(name.replace('\\', '/')).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen file changed: {name}')
    tick = time.perf_counter()
    subprocess.run([sys.executable, 'predict.py', '--method', 'evidence_method.py',
                    '--checkpoint', str(root/'model_bundle'), '--split', 'test',
                    '--output', str(root/'test-predictions.json')], check=True)
    subprocess.run([sys.executable, 'score_predictions.py', '--split', 'test',
                    '--predictions', str(root/'test-predictions.json'),
                    '--output', str(root/'test-score.json')], check=True)
    result = dict(fresh_prediction_plus_scoring_seconds=time.perf_counter()-tick,
                  threads=4, frozen_hashes_verified=True,
                  reference_cpu_verified=False, python=platform.python_version(),
                  platform=platform.platform(), processor=platform.processor())
    (root/'test-timing.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
