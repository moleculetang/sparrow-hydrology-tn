"""Recover a keyed resource token without trusting stale embedded lease PID.

The registry maps token -> lease; an adopted lease's value historically lacked
the token itself. Recover only one exact live child/project/stage match.
"""


def resolve_record_lease(record, leases, project='20261008_1'):
    stage='audit' if record['stage']=='same_point_audit' else record['stage']
    expected=stage+'/'+record['job']
    stored=record['lease']
    token=stored.get('token')
    if token is None:
        matches=[key for key,lease in leases.items()
                 if lease.get('pid')==record['pid'] and lease.get('created')==record['created']
                 and lease.get('project')==project and lease.get('job')==expected]
        if len(matches)!=1:
            raise RuntimeError('EXACT_UNIQUE_KEYED_CHILD_LEASE_REQUIRED')
        token=matches[0]
    lease=leases.get(token)
    if not lease or lease.get('pid')!=record['pid'] or lease.get('created')!=record['created'] or lease.get('project')!=project or lease.get('job')!=expected:
        raise RuntimeError('EXACT_KEYED_CHILD_LEASE_IDENTITY_REQUIRED')
    for key in ('cpus','threads','reserved_bytes','gpu'):
        if stored.get(key)!=lease.get(key):
            raise RuntimeError('KEYED_CHILD_LEASE_RESERVATION_CHANGED '+key)
    return dict(lease,token=token)
