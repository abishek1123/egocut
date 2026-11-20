import itertools, numpy as np
from imp_search import load_summe, evaluate, KEYS
data = load_summe(); rng = np.random.default_rng(0); idx = rng.permutation(len(data)); folds = np.array_split(idx, 5)
grid = [-1, 0, 1, 2]; cands = [(dict(zip(KEYS, ws)), L, sm) for ws in itertools.product(grid, repeat=len(KEYS)) if any(ws) for L in (5, 6, 8) for sm in (1, 5, 9)]
outs = []
for k, te in enumerate(folds):
    tr = [data[i] for i in idx if i not in te]; ts = [data[i] for i in te]
    best = max(cands, key=lambda c: evaluate(tr, *c)); outs.append(evaluate(ts, *best)); print(f'fold {k}: held-out F1 {100*outs[-1]:.1f}  chosen {best}', flush=True)
print(f'5-fold CV held-out F1: {100*np.mean(outs):.1f}')
simple = {'cen': 1, 'nov': 1, 'chg': 0, 'sfn': 1, 'sfc': 1, 'sfnov': 0}
print(f'simple equal-weights config on all SumMe: {100*evaluate(data, simple, 5, 5):.1f}')
