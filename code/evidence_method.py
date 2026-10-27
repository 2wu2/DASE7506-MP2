"""Frozen evidence-augmented or cross-fitted fusion selector."""
import json
from pathlib import Path
import numpy as np
from features import design
from evidence_features import evidence, select_block, NAMES
from robust_method import answer_structure, raw_scores, choose


def score(raw, weights):
    return raw_scores((raw-weights['mean'])/weights['scale'], weights)


def design_variant(base, ev, variant):
    if variant == 'baseline':
        return base
    if variant == 'evidence_only':
        return ev
    block = select_block(ev, variant)
    return np.concatenate([base, block], axis=2)


def fusion_features(base_scores, rule_scores, ev, valid, counts):
    # Center using only valid candidates. Invalid raw logits do not affect
    # confidence features of valid candidates.
    b = base_scores-(base_scores*valid).sum(1, keepdims=True)/np.maximum(1, valid.sum(1, keepdims=True))
    r = rule_scores-(rule_scores*valid).sum(1, keepdims=True)/np.maximum(1, valid.sum(1, keepdims=True))
    checked = ev[:, :, NAMES.index('has_checked_equation')]
    consensus = np.where(valid, counts/4, 0)
    return np.stack([b, r, consensus, b*checked], axis=2)


def load_weights(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    p = Path(checkpoint)
    config = json.loads((p/'config.json').read_text())
    aa, valid, counts = answer_structure(bank)
    base = design(bank, arrays, 'combined').reshape(len(bank), 4, -1)
    ev = evidence(bank) if config['variant'] != 'baseline' else None
    if config['variant'] == 'fusion':
        bs = score(base, load_weights(p/'base.npz'))
        rs = score(ev, load_weights(p/'rules.npz'))
        scores = score(fusion_features(bs, rs, ev, valid, counts), load_weights(p/'meta.npz'))
    else:
        raw = design_variant(base, ev, config['variant'])
        scores = score(raw, load_weights(p/'weights.npz'))
    return choose(scores, aa, valid, counts, grouping=True, mask=True)
