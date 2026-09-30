"""Per-SD happiness weights of the three task-model channels on the Rutledge GBE
gamble data (rutledge_affect_layer.build_rows, unmodified), for the common-scale
check in reviews/CHANNEL_TEST.md section 3.4.
Run from the repo root:  python analysis/channels/rutledge_scale.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import rutledge_affect_layer as ra  # noqa: E402

d = sio.loadmat(str(ROOT / ra.MAT), squeeze_me=True, struct_as_record=False)
sd = d["subjData"]
idx = np.random.RandomState(0).permutation(len(sd))[:ra.MAXSUBJ]
rows = []
for i in idx:
    dd = sd[i].data
    plays = dd if (isinstance(dd, np.ndarray) and dd.dtype == object) else [dd]
    for p in (plays if np.ndim(plays) > 0 else [plays]):
        m = np.asarray(p, float)
        if m.ndim == 2 and m.shape[1] >= 10:
            rows += ra.build_rows(m)
X = np.array([r[0][:3] for r in rows])
y = np.array([r[1] for r in rows])
raw_sd = X.std(0)
w, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), (X - X.mean(0)) / raw_sd]), y, rcond=None)
names = ["fwd", "pres", "back"]
out = dict(n=len(y), raw_sd=dict(zip(names, raw_sd.tolist())), per_sd_weight=dict(zip(names, w[1:].tolist())),
           raw_unit_weight=dict(zip(names, (w[1:] / raw_sd).tolist())))
json.dump(out, open(ROOT / "analysis/channels/out/rutledge_scale.json", "w"), indent=1)
print(out)
