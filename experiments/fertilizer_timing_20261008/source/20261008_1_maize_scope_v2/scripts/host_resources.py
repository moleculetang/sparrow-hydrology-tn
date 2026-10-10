"""Resolve the existing host resource registry, never create a parallel ledger.

Linux requires the explicitly verified shared state path. Windows uses the
native existing default. All lease operations must pass this resolved root.
"""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT.parent/('.compute_coordination' if os.name=='posix' else 'compute_coordination')
if not (MODULE/'resource_registry.py').is_file():raise RuntimeError('EXISTING_HOST_REGISTRY_MODULE_REQUIRED')
sys.path.insert(0,str(MODULE))
import resource_registry as registry


def registry_root():
    if os.name=='nt':return registry.DEFAULT
    value=os.environ.get('D29_RESOURCE_STATE')
    if not value:raise RuntimeError('EXPLICIT_VERIFIED_LINUX_RESOURCE_STATE_REQUIRED')
    root=Path(value).resolve()
    expected=(MODULE/'state').resolve()
    if root!=expected or not (root/'registry.json').is_file():
        raise RuntimeError('VERIFIED_SHARED_LINUX_STATE_REQUIRED_NO_SECOND_LEDGER')
    return root
