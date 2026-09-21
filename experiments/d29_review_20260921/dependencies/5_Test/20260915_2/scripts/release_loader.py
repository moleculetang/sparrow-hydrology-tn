"""Portable snapshot-only read interface. Never falls back to old experiments."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
DEFAULT=Path(__file__).resolve().parents[1]
class InputRelease:
 def __init__(self,root=DEFAULT,allow_sensitivity=False,support_version='frozen_S1_hydrology',training_mode=False):
  if support_version!='frozen_S1_hydrology':raise ValueError('New boundary/source pairing is not validated; cannot silently replace frozen spatial support')
  self.root=Path(root).resolve();self.allow_sensitivity=allow_sensitivity;self.training_mode=training_mode
  self.catalog=json.loads((self.root/'metadata/data_catalog.json').read_text(encoding='utf-8'))
 def path(self,key,verify=True):
  item=self.catalog[key]
  if self.training_mode and item.get('retrospective_cleaning'):raise ValueError('Whole-history archive cleaning is not fold-isolated; use a registered v2 training split')
  if item.get('sensitivity') and not self.allow_sensitivity:raise ValueError('2025 extension requires explicit allow_sensitivity=True')
  p=(self.root/item['path']).resolve()
  if not p.is_relative_to(self.root):raise ValueError('Outside-release path rejected')
  if verify:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
   if h.hexdigest()!=item['sha256']:raise ValueError('Input hash mismatch: '+key)
  return p
 def table(self,key):return pd.read_parquet(self.path(key))
 def array(self,key):return np.load(self.path(key),mmap_mode='r',allow_pickle=False)
 def topology(self):
  """Model-compatible graph with verified, hydrology-aligned support bundle."""
  contract=json.loads(self.path('hydrology_topology_contract').read_text(encoding='utf-8'))
  for name,expected in contract['files'].items():
   p=(self.root/'canonical/spatial'/name).resolve()
   if not p.is_relative_to(self.root/'canonical/spatial'):raise ValueError('Topology bundle path outside release')
   h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
   if h.hexdigest()!=expected:raise ValueError('Mixed/modified topology support: '+name)
  return json.loads(self.path('hydrology_topology').read_text(encoding='utf-8'))
