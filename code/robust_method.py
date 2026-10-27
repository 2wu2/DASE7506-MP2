"""Reference-free CPU selection, with numeric validity and answer grouping."""
import json
from pathlib import Path
import numpy as np
from common import parse_answer
from features import design


def answer_structure(bank):
    answers = [[parse_answer(c['text']) for c in q['candidates']] for q in bank]
    valid = np.array([[a is not None for a in aa] for aa in answers], dtype=bool)
    counts = np.array([[aa.count(a) if a is not None else 1 for a in aa]
                       for aa in answers], dtype=np.float64)
    return answers, valid, counts


def raw_scores(x, weights):
    if 'hidden' not in weights:
        return x @ weights['coef']
    x = np.clip(x, -8, 8)
    h = np.tanh(x @ weights['hidden'] + weights['bias'])
    return h @ weights['output']


def choose(scores, answers, valid, counts, grouping=False, mask=True):
    scores = np.array(scores, copy=True)
    if mask:
        # All-invalid inputs are outside the supplied guarantee; return a
        # deterministic choice without NaN instead of crashing.
        scores[~valid & valid.any(axis=1, keepdims=True)] = -1e30
    if not grouping:
        return scores.argmax(1).tolist()
    result = []
    for s, aa in zip(scores, answers):
        groups = {}
        for j, a in enumerate(aa):
            if a is not None:
                groups.setdefault(a, []).append(j)
        if not groups:
            result.append(int(s.argmax()))
            continue
        group_scores = []
        representatives = []
        for members in groups.values():
            v = s[members]
            # log mean exp: repeated answers do not automatically gain mass.
            top = v.max()
            group_scores.append(top + np.log(np.exp(v-top).mean()))
            representatives.append(members[int(v.argmax())])
        result.append(representatives[int(np.argmax(group_scores))])
    return result


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    p = Path(checkpoint)
    config = json.loads((p/'config.json').read_text())
    answers, valid, counts = answer_structure(bank)
    x = design(bank, arrays, config['mode'])
    with np.load(p/'weights.npz', allow_pickle=False) as archive:
        weights = dict(archive)
    x = ((x-weights['mean'])/weights['scale']).reshape(len(bank), 4, -1)
    scores = raw_scores(x, weights)
    scores += config.get('consensus_weight', 0) * np.where(valid, counts/4, 0)
    return choose(scores, answers, valid, counts, config.get('grouping', False),
                  config.get('mask', True))
