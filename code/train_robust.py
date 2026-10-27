"""Predeclared 3-fold, question-level CV. Test files are never read.

Hyperparameters are selected on out-of-fold training predictions. Validation
is evaluated once after the selected configuration and weights are saved.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from common import load_inputs
from features import design
from train import labels
from optimize_hard import masks
from robust_method import answer_structure, raw_scores, choose, predict

OUT = Path('runs/robust_v3')
SEED = 20261027
CONFIGS = [
    dict(name='v1_original', family='ridge', mode='embedding', alpha=100, boost=1,
         mixed_only=False, consensus_weight=.25, grouping=False, mask=False),
    dict(name='v1_valid', family='ridge', mode='embedding', alpha=100, boost=1,
         mixed_only=False, consensus_weight=.25, grouping=False, mask=True),
    dict(name='v2_original', family='ridge', mode='combined', alpha=1000, boost=3,
         mixed_only=True, consensus_weight=0, grouping=False, mask=False),
    dict(name='ridge_uniform', family='ridge', mode='combined', alpha=1000, boost=1,
         mixed_only=True, consensus_weight=0, grouping=False, mask=True),
    dict(name='linear_group_003', family='listwise', mode='combined', reg=.03,
         grouping=True, mask=True, hidden=0, epochs=80),
    dict(name='linear_group_01', family='listwise', mode='combined', reg=.1,
         grouping=True, mask=True, hidden=0, epochs=80),
    dict(name='mlp32_group_001', family='listwise', mode='combined', reg=.001,
         grouping=True, mask=True, hidden=32, epochs=60),
    dict(name='mlp32_group_01', family='listwise', mode='combined', reg=.01,
         grouping=True, mask=True, hidden=32, epochs=60),
]


def evaluate(y, choices, hard, minority):
    ok = y[np.arange(len(y)), choices].astype(bool)
    result = {}
    for name, m in [('overall', np.ones(len(y), dtype=bool)), ('hard', hard), ('minority', minority)]:
        result[name] = dict(n=int(m.sum()), correct=int(ok[m].sum()),
                            accuracy=float(ok[m].mean()*100) if m.any() else None)
    return result


def fit(raw, y, valid, counts, minority, config):
    mean, scale = raw.mean(axis=(0, 1)), raw.std(axis=(0, 1))
    scale[scale < 1e-8] = 1
    x = (raw-mean)/scale
    if config['family'] == 'ridge':
        xc = x-x.mean(1, keepdims=True)
        target = y-y.mean(1, keepdims=True)
        qw = 1+(config['boost']-1)*minority.astype(float)
        if config['mixed_only']:
            qw *= y.min(1) != y.max(1)
        xx = xc.reshape(-1, x.shape[-1]); ww = np.repeat(qw, 4)
        # Keep regularization per training question stable across fold sizes.
        alpha = config['alpha'] * len(y)/4755
        coef = np.linalg.solve(xx.T@(xx*ww[:, None])+alpha*np.eye(xx.shape[1]),
                               xx.T@(target.ravel()*ww))
        return dict(mean=mean, scale=scale, coef=coef)

    # If every valid answer is correct, both log partitions coincide and the
    # selection loss is zero; such questions can be omitted exactly.
    informative = ((~y.astype(bool)) & valid).any(1)
    x, y, valid, counts = x[informative], y[informative], valid[informative], counts[informative]
    x = x.astype(np.float32)
    rng = np.random.default_rng(SEED)
    hidden = config['hidden']
    if hidden:
        x = np.clip(x, -8, 8)
        params = dict(hidden=(rng.normal(size=(x.shape[-1], hidden))/np.sqrt(x.shape[-1])).astype(np.float32),
                      bias=np.zeros(hidden, np.float32), output=np.zeros(hidden, np.float32))
    else:
        params = dict(coef=np.zeros(x.shape[-1], np.float32))
    mom = {k: np.zeros_like(v) for k, v in params.items()}
    vel = {k: np.zeros_like(v) for k, v in params.items()}
    average = {k: np.zeros_like(v) for k, v in params.items()}
    count = 0; step = 0
    for epoch in range(config['epochs']):
        order = rng.permutation(len(y))
        for offset in range(0, len(order), 128):
            ix = order[offset:offset+128]; bx = x[ix]
            if hidden:
                h = np.tanh(bx @ params['hidden'] + params['bias'])
                score = h @ params['output']
            else:
                score = bx @ params['coef']
            score = score.astype(np.float64)-np.log(counts[ix])
            score[~valid[ix]] = -1e30
            e = np.exp(score-score.max(1, keepdims=True))
            all_p = e/e.sum(1, keepdims=True)
            correct_e = e*y[ix]
            correct_p = correct_e/np.maximum(correct_e.sum(1, keepdims=True), 1e-100)
            ds = ((all_p-correct_p)/len(ix)).astype(np.float32)
            if hidden:
                dh = ds[:, :, None]*params['output']*(1-h*h)
                grads = dict(output=np.einsum('bch,bc->h', h, ds),
                             hidden=bx.reshape(-1, bx.shape[-1]).T @ dh.reshape(-1, hidden),
                             bias=dh.sum((0, 1)))
            else:
                grads = dict(coef=bx.reshape(-1, bx.shape[-1]).T @ ds.ravel())
            step += 1
            for key in params:
                # Biases are not regularized. Objective is mean question loss.
                g = grads[key] + (0 if key == 'bias' else config['reg']*params[key])
                mom[key] = .9*mom[key]+.1*g
                vel[key] = .999*vel[key]+.001*g*g
                params[key] -= .003*(mom[key]/(1-.9**step))/(np.sqrt(vel[key]/(1-.999**step))+1e-8)
        # Predeclared final-10-epoch averaging, no fold-label early stopping.
        if epoch >= config['epochs']-10:
            for key in params:
                average[key] += params[key]
            count += 1
    return dict(mean=mean, scale=scale, **{k: v/count for k, v in average.items()})


def selections(raw, weights, config, answers, valid, counts):
    x = (raw-weights['mean'])/weights['scale']
    scores = raw_scores(x, weights)
    scores += config.get('consensus_weight', 0)*np.where(valid, counts/4, 0)
    return choose(scores, answers, valid, counts, config['grouping'], config['mask'])


def main():
    tick = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(seed=SEED, folds=3, configs=CONFIGS,
                    selection='pooled train OOF hard accuracy, then overall; ties retain earlier config',
                    validation='evaluate once after CV selection and full-train refit',
                    test='never accessed', tuning_after_validation=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    bank, arrays = load_inputs('train'); y = labels(bank, 'train')
    hard, minority = masks(bank, y)
    answers, valid, counts = answer_structure(bank)
    raws = {mode: design(bank, arrays, mode).reshape(len(bank), 4, -1)
            for mode in ('embedding', 'combined')}
    # Stratify entire questions; IDs are never model features.
    strata = y.sum(1)*4+hard.astype(int)*2+minority.astype(int)
    folds = np.empty(len(y), dtype=int); rng = np.random.default_rng(SEED)
    for value in np.unique(strata):
        ix = rng.permutation(np.flatnonzero(strata == value))
        folds[ix] = np.arange(len(ix)) % 3
    np.save(OUT/'train-folds.npy', folds)
    results = []; best = None; best_key = (-1, -1)
    for config in CONFIGS:
        ct = time.perf_counter(); oof = np.empty(len(y), dtype=int); per_fold = []
        for fold in range(3):
            tr = folds != fold; ev = ~tr
            weights = fit(raws[config['mode']][tr], y[tr], valid[tr], counts[tr], minority[tr], config)
            picks = selections(raws[config['mode']][ev], weights, config,
                               [a for a, keep in zip(answers, ev) if keep], valid[ev], counts[ev])
            oof[ev] = picks
            per_fold.append(evaluate(y[ev], picks, hard[ev], minority[ev]))
            print(config['name'], 'fold', fold, per_fold[-1], flush=True)
        summary = evaluate(y, oof, hard, minority)
        record = dict(config=config, metrics=summary, per_fold=per_fold, seconds=time.perf_counter()-ct)
        results.append(record)
        np.save(OUT/(config['name']+'-oof.npy'), oof)
        (OUT/'cv-results.json').write_text(json.dumps(results, indent=2))
        key = (summary['hard']['accuracy'], summary['overall']['accuracy'])
        if key > best_key:
            best_key = key; best = config
    # Freeze the model before loading validation inputs or references.
    weights = fit(raws[best['mode']], y, valid, counts, minority, best)
    bundle = OUT/'model_bundle'; bundle.mkdir(exist_ok=True)
    np.savez(bundle/'weights.npz', **weights)
    (bundle/'config.json').write_text(json.dumps(best, indent=2))
    files = [Path('robust_method.py'), Path('features.py'), Path('train_robust.py'), OUT/'protocol.json']+list(bundle.iterdir())
    (OUT/'frozen.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2))
    vb, va = load_inputs('validation'); vy = labels(vb, 'validation'); vh, vm = masks(vb, vy)
    vp = predict(vb, va, bundle)
    result = dict(selected=best, cv=best_key, validation=evaluate(vy, vp, vh, vm),
                  total_seconds=time.perf_counter()-tick, seed=SEED, threads=4,
                  asset_bytes=sum(p.stat().st_size for p in bundle.iterdir()), test_accessed=False)
    (OUT/'summary.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
