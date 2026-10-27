"""Train on train only, select on validation only. Deterministic, no randomness."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='4'
import json,time,platform,hashlib
from pathlib import Path
import numpy as np
from common import DATA,load_inputs,read_jsonl,parse_answer
from features import design,features

def labels(bank,split):
    refs=read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    assert [r['id'] for r in refs]==[r['id'] for r in bank]
    return np.array([[int(parse_answer(c['text']) is not None and parse_answer(c['text'])==parse_answer(r['answer'])) for c in q['candidates']] for q,r in zip(bank,refs)])
def acc(y,c): return float(100*y[np.arange(len(y)),c].mean())
def main():
    tick=time.perf_counter(); root=Path('runs'); root.mkdir(exist_ok=True)
    tb,ta=load_inputs('train'); vb,va=load_inputs('validation')
    yt,yv=labels(tb,'train'),labels(vb,'validation')
    consensus=features(vb)[:,1].reshape(-1,4)
    records=[dict(mode='first',validation_accuracy=acc(yv,np.zeros(len(vb),dtype=int))),dict(mode='majority',validation_accuracy=acc(yv,consensus.argmax(1)))]
    best=-1
    for mode in ('text','embedding','combined'):
        x,v=design(tb,ta,mode),design(vb,va,mode)
        mean,scale=x.mean(0),x.std(0); scale[scale<1e-8]=1
        x=(x-mean)/scale; v=(v-mean)/scale
        shape=x.shape; grouped=x.reshape(-1,4,shape[1]); x=(grouped-grouped.mean(1,keepdims=True)).reshape(shape)
        target=(yt-yt.mean(1,keepdims=True)).ravel()
        gram,rhs=x.T@x,x.T@target
        for alpha in (1,10,100,1000):
            coef=np.linalg.solve(gram+alpha*np.eye(shape[1]),rhs)
            s=(v@coef).reshape(-1,4)
            for weight in (0,.1,.25,.5,1):
                score=acc(yv,(s+weight*consensus).argmax(1))
                rec=dict(mode=mode,alpha=alpha,consensus_weight=weight,validation_accuracy=score); records.append(rec)
                if score>best:
                    best=score; p=root/'model_bundle'; p.mkdir(exist_ok=True)
                    np.savez(p/'weights.npz',mean=mean,scale=scale,coef=coef)
                    (p/'config.json').write_text(json.dumps(rec,indent=2))
        print(mode,max(r['validation_accuracy'] for r in records if r['mode']==mode),flush=True)
    assets=list((root/'model_bundle').iterdir())
    report=dict(seed=0,training_seconds=time.perf_counter()-tick,python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),processor=platform.processor(),threads=4,experiments=records,assets_bytes=sum(p.stat().st_size for p in assets),asset_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in assets})
    (root/'experiments.json').write_text(json.dumps(report,indent=2))
    print('best validation',best,flush=True)
if __name__=='__main__': main()
