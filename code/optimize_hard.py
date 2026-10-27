"""Hard-question optimization: train/validation only; never load test."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='4'
import json,time,hashlib
from pathlib import Path
import numpy as np
from common import load_inputs,parse_answer
from features import design,features
from train import labels
from submission import predict as baseline_predict

def masks(bank,y):
    hard=[]; minority=[]
    for q,yy in zip(bank,y):
        aa=[parse_answer(c['text']) for c in q['candidates']]
        hard.append(len({a for a in aa if a is not None})>=2)
        freq=[aa.count(a) if a is not None else 0 for a in aa]
        minority.append(max([f for f,z in zip(freq,yy) if z],default=0)<max(freq))
    return np.array(hard),np.array(minority)

def metrics(y,p,hard,minority,base):
    ok=y[np.arange(len(y)),p].astype(bool); old=y[np.arange(len(y)),base].astype(bool)
    return dict(accuracy=float(ok.mean()*100),hard_accuracy=float(ok[hard].mean()*100),minority_accuracy=float(ok[minority].mean()*100),hard_n=int(hard.sum()),minority_n=int(minority.sum()),fixed=int((ok&~old).sum()),regressed=int((~ok&old).sum()))

def main():
    tick=time.perf_counter(); out=Path('runs/hard_v2'); out.mkdir(exist_ok=True)
    tb,ta=load_inputs('train'); vb,va=load_inputs('validation')
    yt,yv=labels(tb,'train'),labels(vb,'validation'); ht,mt=masks(tb,yt); hv,mv=masks(vb,yv)
    base=np.array(baseline_predict(vb,va,Path('runs/baseline_v1/model_bundle')))
    baseline=metrics(yv,base,hv,mv,base); records=[]; bestkey=(baseline['hard_accuracy'],baseline['accuracy'],baseline['minority_accuracy'])
    best=None
    cons=features(vb)[:,1].reshape(-1,4)
    for mode in ('embedding','combined'):
        raw=design(tb,ta,mode); vr=design(vb,va,mode)
        mean=raw.mean(0); scale=raw.std(0); scale[scale<1e-8]=1
        x=((raw-mean)/scale).reshape(-1,4,raw.shape[1]); v=((vr-mean)/scale).reshape(-1,4,raw.shape[1])
        xc=x-x.mean(1,keepdims=True); target=yt-yt.mean(1,keepdims=True)
        mixed=(yt.min(1)!=yt.max(1))
        for boost in (1,3):
            qw=mixed.astype(float)*(1+(boost-1)*mt)
            xx=xc.reshape(-1,raw.shape[1]); ww=np.repeat(qw,4)
            gram=xx.T@(xx*ww[:,None]); rhs=xx.T@(target.ravel()*ww)
            for alpha in (10,100,1000):
                coef=np.linalg.solve(gram+alpha*np.eye(raw.shape[1]),rhs)
                scores=v@coef
                for cw in (0,.1,.25):
                    p=(scores+cw*cons).argmax(1); m=metrics(yv,p,hv,mv,base)
                    rec=dict(family='weighted_ridge',mode=mode,boost=boost,alpha=alpha,consensus_weight=cw,**m);records.append(rec)
                    key=(m['hard_accuracy'],m['accuracy'],m['minority_accuracy'])
                    if key>bestkey:
                        bestkey=key;best=(rec,mean.copy(),scale.copy(),coef.copy())
        # Correct-vs-wrong pairs; equal total weight per question.
        diffs=[]; qids=[]; weights=[]
        for i in np.flatnonzero(mixed):
            pos=np.flatnonzero(yt[i]); neg=np.flatnonzero(1-yt[i])
            for c in pos:
                for w in neg:
                    diffs.append(x[i,c]-x[i,w]);qids.append(i);weights.append(1/(len(pos)*len(neg)))
        d=np.array(diffs); qids=np.array(qids); weights=np.array(weights)
        for boost in (1,3):
            pw=weights*(1+(boost-1)*mt[qids]); pw/=pw.sum()
            for reg in (.01,.1):
                coef=np.zeros(d.shape[1]); mom=np.zeros_like(coef); vel=np.zeros_like(coef)
                for step in range(1,301):
                    margin=np.clip(d@coef,-40,40)
                    grad=-(d.T@(pw/(1+np.exp(margin))))+reg*coef
                    mom=.9*mom+.1*grad;vel=.999*vel+.001*grad*grad
                    coef-=.01*(mom/(1-.9**step))/(np.sqrt(vel/(1-.999**step))+1e-8)
                scores=v@coef
                for cw in (0,.5,1):
                    p=(scores+cw*cons).argmax(1);m=metrics(yv,p,hv,mv,base)
                    rec=dict(family='pairwise_logistic',mode=mode,boost=boost,regularization=reg,steps=300,consensus_weight=cw,**m); records.append(rec)
                    key=(m['hard_accuracy'],m['accuracy'],m['minority_accuracy'])
                    if key>bestkey:
                        bestkey=key;best=(rec,mean.copy(),scale.copy(),coef.copy())
        print(mode,'best',bestkey,flush=True)
    if best:
        rec,mean,scale,coef=best; p=out/'model_bundle';p.mkdir(exist_ok=True)
        np.savez(p/'weights.npz',mean=mean,scale=scale,coef=coef)
        (p/'config.json').write_text(json.dumps(rec,indent=2))
    else: rec=None
    report=dict(baseline=baseline,selected=rec,selection_rule='lexicographic hard accuracy, overall accuracy, minority accuracy; validation only',train_mixed=int(mixed.sum()),train_minority=int(mt.sum()),experiments=records,seconds=time.perf_counter()-tick,seed=0,threads=4,test_accessed=False)
    (out/'experiments.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='experiments'}),flush=True)
if __name__=='__main__':main()
