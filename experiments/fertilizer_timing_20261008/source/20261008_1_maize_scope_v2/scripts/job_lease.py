"""Verify the exact host-local stage lease before scientific side effects."""
import os
from host_resources import registry,registry_root


def require(jid,stage):
    token=os.environ.get('D29_LEASE_TOKEN')
    lease=registry.status(root=registry_root())['leases'].get(token)
    if not lease or lease['project']!='20261008_1' or lease['job']!=stage+'/'+jid:
        raise RuntimeError('EXACT_SCIENTIFIC_STAGE_LEASE_REQUIRED')
    if lease['pid']!=os.getpid() or lease['created']!=registry.identity(os.getpid()):
        raise RuntimeError('EXACT_LIVE_STAGE_OWNER_REQUIRED')
    if lease['threads']!=int(os.environ.get('D29_WORKER_THREADS','1')):
        raise RuntimeError('RESERVED_STAGE_THREADS_MISMATCH')
    return lease
