"""All evidence directions with outer CV and nested cross-fitting for fusion.

Only supplied train references are fitted. Validation is opened after freeze.
No test data or test scores are opened.
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '4'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from common import load_inputs
from features import design
from train import labels
from train_robust import fit, evaluate
from optimize_hard import masks
from robust_method import answer_structure, choose
from evidence_features import evidence, NAMES
from evidence_method import score, design_variant, fusion_features, predict

OUT = Path('runs/evidence_v4')
SEED = 20261027
TRAIN_CONFIG = dict(family='listwise', mode='combined', reg=.03,
                    grouping=True, mask=True, hidden=0, epochs=80)
VARIANTS = ['baseline', 'arithmetic', 'consistency', 'constraints', 'all', 'evidence_only']


def folds_for(y, hard, minority):
    strata = y.sum(1)*4+hard.astype(int)*2+minority.astype(int)
    folds = np.empty(len(y), dtype=int); rng = np.random.default_rng(SEED)
    for v in np.unique(strata):
        ix = rng.permutation(np.flatnonzero(strata == v))
        folds[ix] = np.arange(len(ix)) % 3
    return folds


def picks(scores, answers, valid, counts):
    return choose(scores, answers, valid, counts, grouping=True, mask=True)


def main():
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(variants=VARIANTS+['fusion'], seed=SEED, outer_folds=3, fusion_inner_folds=3,
                    fixed_training=TRAIN_CONFIG,
                    selection='beats baseline pooled hard accuracy, improves hard accuracy in at least 2 folds; rank by pooled hard then overall',
                    fallback='baseline if no qualifying improvement',
                    validation='one evaluation after configuration and assets frozen',
                    test_access='none', new_search_after_validation=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    bank, arrays = load_inputs('train'); y = labels(bank, 'train')
    hard, minority = masks(bank, y); aa, valid, counts = answer_structure(bank)
    base = design(bank, arrays, 'combined').reshape(len(bank), 4, -1)
    coverage = json.loads((OUT/'feature-coverage.json').read_text())
    digest = hashlib.sha256(Path('evidence_features.py').read_bytes()).hexdigest()
    if digest != coverage['source_sha256']:
        raise RuntimeError('Evidence cache was built with different feature source')
    with np.load(OUT/'train-evidence-cache.npz', allow_pickle=False) as cache:
        ev = cache['features'].copy()
    assert ev.shape == (len(bank), 4, len(NAMES))
    folds = folds_for(y, hard, minority); np.save(OUT/'train-folds.npy', folds)
    oof_scores = {}; results = []
    for variant in VARIANTS:
        tick = time.perf_counter(); raw = design_variant(base, ev, variant)
        oof = np.empty((len(y), 4)); per_fold = []
        for fold in range(3):
            tr = folds != fold; val = ~tr
            weights = fit(raw[tr], y[tr], valid[tr], counts[tr], minority[tr], TRAIN_CONFIG)
            oof[val] = score(raw[val], weights)
            choices = picks(oof[val], [a for a, use in zip(aa, val) if use], valid[val], counts[val])
            per_fold.append(evaluate(y[val], choices, hard[val], minority[val]))
            print(variant, 'fold', fold, 'hard', per_fold[-1]['hard']['accuracy'], flush=True)
        oof_scores[variant] = oof
        choices = picks(oof, aa, valid, counts)
        np.save(OUT/(variant+'-oof-scores.npy'), oof)
        record = dict(variant=variant, metrics=evaluate(y, choices, hard, minority),
                      per_fold=per_fold, seconds=time.perf_counter()-tick)
        results.append(record)
        (OUT/'cv-results.json').write_text(json.dumps(results, indent=2))

    # Outer evaluation of stacking. Outer validation labels never participate
    # in either base fitting or meta fitting. Inner OOF makes meta examples.
    tick = time.perf_counter(); fusion_oof = np.empty((len(y), 4)); per_fold = []
    for fold in range(3):
        tr = folds != fold; val = ~tr
        inner = folds_for(y[tr], hard[tr], minority[tr])
        inner_base = np.empty((tr.sum(), 4)); inner_rules = np.empty_like(inner_base)
        for j in range(3):
            it = inner != j; iv = ~it
            for raw, target in [(base[tr], inner_base), (ev[tr], inner_rules)]:
                w = fit(raw[it], y[tr][it], valid[tr][it], counts[tr][it], minority[tr][it], TRAIN_CONFIG)
                target[iv] = score(raw[iv], w)
        meta_raw = fusion_features(inner_base, inner_rules, ev[tr], valid[tr], counts[tr])
        meta_w = fit(meta_raw, y[tr], valid[tr], counts[tr], minority[tr], TRAIN_CONFIG)
        # These predictions come from outer training fits above, never from
        # any fit which has seen the outer validation labels.
        meta_val = fusion_features(oof_scores['baseline'][val], oof_scores['evidence_only'][val],
                                   ev[val], valid[val], counts[val])
        fusion_oof[val] = score(meta_val, meta_w)
        choices = picks(fusion_oof[val], [a for a, use in zip(aa, val) if use], valid[val], counts[val])
        per_fold.append(evaluate(y[val], choices, hard[val], minority[val]))
        print('nested_fusion fold', fold, 'hard', per_fold[-1]['hard']['accuracy'], flush=True)
    fusion_choices = picks(fusion_oof, aa, valid, counts)
    results.append(dict(variant='fusion', metrics=evaluate(y, fusion_choices, hard, minority),
                        per_fold=per_fold, seconds=time.perf_counter()-tick))
    oof_scores['fusion'] = fusion_oof
    np.save(OUT/'fusion-oof-scores.npy', fusion_oof)
    (OUT/'cv-results.json').write_text(json.dumps(results, indent=2))

    reference = results[0]; selected = reference
    for result in results[1:]:
        wins = sum(a['hard']['accuracy'] > b['hard']['accuracy']
                   for a, b in zip(result['per_fold'], reference['per_fold']))
        result['folds_improved_over_baseline'] = wins
        key = (result['metrics']['hard']['accuracy'], result['metrics']['overall']['accuracy'])
        old = (selected['metrics']['hard']['accuracy'], selected['metrics']['overall']['accuracy'])
        if wins >= 2 and result['metrics']['hard']['accuracy'] > reference['metrics']['hard']['accuracy'] and key > old:
            selected = result
    variant = selected['variant']
    bundle = OUT/'model_bundle'; bundle.mkdir(exist_ok=True)
    if variant == 'fusion':
        base_w = fit(base, y, valid, counts, minority, TRAIN_CONFIG)
        rule_w = fit(ev, y, valid, counts, minority, TRAIN_CONFIG)
        meta_raw = fusion_features(oof_scores['baseline'], oof_scores['evidence_only'], ev, valid, counts)
        meta_w = fit(meta_raw, y, valid, counts, minority, TRAIN_CONFIG)
        for name, w in [('base', base_w), ('rules', rule_w), ('meta', meta_w)]:
            np.savez(bundle/(name+'.npz'), **w)
    else:
        weights = fit(design_variant(base, ev, variant), y, valid, counts, minority, TRAIN_CONFIG)
        np.savez(bundle/'weights.npz', **weights)
    config = dict(variant=variant, training=TRAIN_CONFIG, seed=SEED)
    (bundle/'config.json').write_text(json.dumps(config, indent=2))
    (OUT/'cv-results.json').write_text(json.dumps(results, indent=2))
    files = [Path(p) for p in ['evidence_method.py', 'evidence_features.py', 'features.py',
             'robust_method.py', 'train_evidence.py', 'train_robust.py']]+[OUT/'protocol.json']+list(bundle.iterdir())
    (OUT/'frozen.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2))
    # Freeze completed. The official validation bank is only now loaded.
    vb, va = load_inputs('validation'); vy = labels(vb, 'validation'); vh, vm = masks(vb, vy)
    prediction = predict(vb, va, bundle)
    result = dict(selected=variant, cv=selected, validation=evaluate(vy, prediction, vh, vm),
                  total_training_seconds=time.perf_counter()-started, feature_seconds=coverage['seconds'],
                  asset_bytes=sum(p.stat().st_size for p in bundle.iterdir()),
                  seed=SEED, threads=4, test_accessed=False)
    (OUT/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
