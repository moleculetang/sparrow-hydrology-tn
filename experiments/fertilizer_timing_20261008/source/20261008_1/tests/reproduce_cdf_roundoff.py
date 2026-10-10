"""Saved failing interseason target case, independent of any fitted TN."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from timing.profiles import profile

profiles=json.loads((ROOT/'inputs/calendar_support/reach_profiles.json').read_text())
failures=[];worst=0.
for year in (2016,2021,2024):
    for crop,rs in profiles.items():
        for reach,p in rs.items():
            _,_,g,o,shape=profile(year,p['seasons'],return_target_shape=True)
            oldg=np.cumsum(g);oldo=np.cumsum(o)
            oldg[np.flatnonzero(g>0)[-1]:]=1.;oldo[np.flatnonzero(o>0)[-1]:]=1.
            old=oldg-oldo
            if np.any(old<0):
                day=int(np.argmin(old));failures.append(dict(year=year,crop=crop,reach=int(reach),day=day,
                    old_target_per_kg=float(old[day]),new_target_per_kg=float(shape[day])))
            assert np.min(shape)>=0 and shape[-1]==0
            worst=min(worst,float(old.min()))
assert failures,'FAILURE_MUST_BE_ACTUALLY_REPRODUCED'
receipt=dict(old_failure_reproduced=True,failures=len(failures),examples=failures[:10],
  old_minimum_per_kg=worst,new_negative_targets=0,stock_clipping=False,
  correction='season-specific completed CDF identities before season aggregation',NSE='not applicable')
(ROOT/'evidence/cdf_roundoff_failure_and_fix.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
