"""Durable worker state, immutable identity and full RNG restoration."""
import hashlib
import os
import random
import time
import numpy as np
import torch
from fc_io import RUN,local,replace,read,write,sha,now,memory


def save(path,state):
    path=local(path)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    with temp.open('wb') as stream:
        torch.save(state,stream)
        stream.flush();os.fsync(stream.fileno())
    replace(temp,path)
    return sha(path)


def load(path,identity):
    # Only campaign-created checkpoints are accepted, paired with immutable
    # manifest identity. This is not a loader for arbitrary external pickle.
    path=local(path)
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value['identity']!=identity:raise RuntimeError('CHECKPOINT_IDENTITY_MISMATCH')
    return value


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if value['cuda']:torch.cuda.set_rng_state_all(value['cuda'])


class ResourceYield(Exception):pass
class BudgetStop(Exception):pass


class WorkerGuard:
    def __init__(self,tag,deadline=None,device='cpu'):
        self.tag=tag;self.deadline=deadline;self.device=device
        self.peak_ram_percent=0.;self.peak_vram_percent=0.
        self.started_cpu=time.process_time();self.started_wall=time.time()

    def check(self):
        if self.deadline is not None and time.time()>=self.deadline:raise BudgetStop('TRAINING_DEADLINE_REACHED')
        request=RUN/'work/requests'/f'{self.tag}.json'
        if request.exists():
            value=read(request)
            if value['action']=='budget_stop':raise BudgetStop('CONTROLLER_BUDGET_STOP')
            if value['action']=='resource_yield':raise ResourceYield('CONTROLLER_RESOURCE_YIELD')
            raise RuntimeError('Unknown controller request')
        ram=memory();self.peak_ram_percent=max(self.peak_ram_percent,ram['used_percent'])
        if ram['used_percent']>=90:raise ResourceYield('SYSTEM_RAM_90_PERCENT_YIELD')
        if self.device.startswith('cuda'):
            free,total=torch.cuda.mem_get_info(self.device)
            used=100*(total-free)/total;self.peak_vram_percent=max(self.peak_vram_percent,used)
            if used>=73:raise ResourceYield('SYSTEM_VRAM_YIELD')
        return ram
