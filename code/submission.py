"""CPU-only candidate selector. Uses no references or IDs."""
import json
from pathlib import Path
import numpy as np
from features import design,features

def predict(bank,arrays,checkpoint):
    if not bank: return []
    p=Path(checkpoint)
    cfg=json.loads((p/'config.json').read_text())
    with np.load(p/'weights.npz',allow_pickle=False) as w:
        x=design(bank,arrays,cfg['mode'])
        s=(((x-w['mean'])/w['scale'])@w['coef']).reshape(-1,4)
    s+=cfg['consensus_weight']*features(bank)[:,1].reshape(-1,4)
    return s.argmax(1).tolist()
