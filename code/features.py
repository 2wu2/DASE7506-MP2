import re
import numpy as np
from common import parse_answer

def features(bank):
    rows=[]
    for q in bank:
        answers=[parse_answer(c['text']) for c in q['candidates']]
        for c,a in zip(q['candidates'],answers):
            t=c['text']
            rows.append([a is not None, sum(b==a for b in answers)/4 if a is not None else 0, np.log1p(len(t)), np.log1p(len(t.split())), np.log1p(len(re.findall(r'\d+',t))), np.log1p(t.count('\n')), np.log1p(t.count('=')), t.count('####'),t.count('\\boxed'),'wait' in t.lower(),'check' in t.lower(),'?' in t,'/' in (a or '')])
    return np.asarray(rows,dtype=np.float64)

def design(bank,arrays,mode):
    if mode=='text': return features(bank)
    e=arrays['embeddings'].reshape(-1,1024).astype(np.float64)
    if mode=='embedding': return e
    return np.concatenate([e,features(bank)],axis=1)
