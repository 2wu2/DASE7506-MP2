import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='4'
import json,time,hashlib
from pathlib import Path
import numpy as np
from common import load_inputs
from evidence_features import evidence,NAMES,checked_equations
out=Path('runs/evidence_v4'); tick=time.perf_counter();bank,a=load_inputs('train');x=evidence(bank)
np.savez(out/'train-evidence-cache.npz',features=x)
report={'seconds':time.perf_counter()-tick,'names':NAMES,'source_sha256':hashlib.sha256(Path('evidence_features.py').read_bytes()).hexdigest(),'candidate_count':len(bank)*4,'any_checked':int((x[:,:,NAMES.index('has_checked_equation')]>0).sum()),'any_wrong':int((x[:,:,NAMES.index('any_wrong')]>0).sum()),'final_differs':int((x[:,:,NAMES.index('last_equation_differs_answer')]>0).sum())}
(out/'feature-coverage.json').write_text(json.dumps(report,indent=2))
rng=np.random.default_rng(2026); sample=[]
for i in rng.permutation(len(bank)):
 for c in bank[i]['candidates']:
  _,eq=checked_equations(c['text'])
  for e in eq:
   if e['wrong'] and len(sample)<12:sample.append({'question':bank[i]['question'],'equation':e,'text':c['text']})
 if len(sample)>=12:break
(out/'equation-audit.json').write_text(json.dumps(sample,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report));print(json.dumps([{'equation':s['equation'],'text':s['text'][:900]} for s in sample[:6]],ensure_ascii=True))
