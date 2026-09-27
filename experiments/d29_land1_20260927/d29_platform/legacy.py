"""Frozen LAND0 imports. All writable roots and compilation caches are new."""
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
from .runtime import ROOT,configure
SNAP=ROOT/'vendor/legacy22'

def bootstrap():
    configure()
    sys.path.insert(0,str(SNAP/'scripts'))
    import campaign_model as cm
    # The snapshot's original import configures its own (new) work cache.
    configure()
    import numba,torch
    numba.config.CACHE_DIR=str(ROOT/'cache/numba')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    return cm

def build_legacy(fold='F23',structure='U',inference_only=False):
    import numpy as np,pandas as pd
    cm=bootstrap()
    a=json.loads((SNAP/'input_potential_v2/configs/anchors.json').read_text(encoding='utf-8'))[fold+'_'+structure]
    design=json.loads((SNAP/'data/designs'/f"{a['fold']}.json").read_text(encoding='utf-8'))
    train=None if inference_only else pd.read_parquet(SNAP/'data/folds'/a['fold']/'train.parquet')
    model=cm.make_model(cm.load_data('FULL24C',verify=True),train,a['kind'],design)
    return model,np.asarray(a['parameters'],dtype=np.float64),a

def load_inverse_checkpoint(structure,resolution):
    import numpy as np
    bootstrap()
    old=Path('E:/SPARROW/5_Test/20260924_1/structure_audit')
    spec=importlib.util.spec_from_file_location('_frozen_inverse',old/'scripts/inverse.py')
    mod=importlib.util.module_from_spec(spec)
    source=(old/'scripts/inverse.py').read_text(encoding='utf-8')
    # Relocate only dependency/data root before executing module imports.
    source=source.replace("OLD = Path('E:/SPARROW/5_Test/20260922_1')",f"OLD = Path({str(SNAP)!r})")
    exec(compile(source,str(old/'scripts/inverse.py'),'exec'),mod.__dict__)
    mod.OLD=SNAP;mod.V2=SNAP/'input_potential_v2'
    model=mod.Inverse(structure,resolution,True,4.)
    folder=old/'reconstruction'/f'{structure}_{resolution}_joint_b4'
    x=np.load(folder/'best.npy',allow_pickle=False)
    meta={'tag':folder.name,'checkpoint':str(folder/'best.npy'),'diagnostic_only':True}
    return model,x,meta
