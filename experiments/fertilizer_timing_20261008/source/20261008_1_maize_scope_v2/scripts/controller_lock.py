"""Lifetime host-local controller lock; never touches scientific child state."""
import os


def acquire(path):
    path.parent.mkdir(parents=True,exist_ok=True);handle=path.open('a+b')
    try:
        if os.name=='posix':
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        else:
            import msvcrt
            if handle.seek(0,2)==0:handle.write(b'0');handle.flush()
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    except OSError as exc:
        handle.close();raise RuntimeError('CONTROLLER_LIFETIME_LOCK_ALREADY_HELD') from exc
    return handle
