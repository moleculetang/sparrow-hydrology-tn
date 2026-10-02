"""Atomic epoch checkpoints with RNG continuity and explicit identity validation."""
import os,random
from pathlib import Path
import numpy as np,torch

def atomic_save(value,path):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(value,tmp)
    os.replace(tmp,path)

def save_epoch(path,model,optimizer,identity,epoch,history,best,selected,stall):
    atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),identity=identity,
        epoch=epoch,history=history,best=best,selected=selected,stall=stall,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        numpy_rng=np.random.get_state(),python_rng=random.getstate()),path)

def restore_epoch(path,model,optimizer,identity,device):
    path=Path(path)
    if not path.exists():return None
    state=torch.load(path,map_location=device,weights_only=False)
    if state.get('identity')!=identity:raise RuntimeError('RESUME_IDENTITY_MISMATCH')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if state['cuda_rng'] and torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    return state
