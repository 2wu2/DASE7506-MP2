"""Verify the delivered package against its SHA256 manifest (stdlib only)."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'))
    failed = []
    for name, expected in manifest.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failed.append(name)
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            failed.append(name)
    if failed:
        raise SystemExit('Package verification failed: '+', '.join(failed))
    print(f'PASS: {len(manifest)} delivered files match SHA256SUMS.json')


if __name__ == '__main__':
    main()
