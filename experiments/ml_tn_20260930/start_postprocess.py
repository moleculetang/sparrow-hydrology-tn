import os,subprocess
from mltn.common import ROOT,read,write
from mltn.resources import registry
path=ROOT/'outputs/postprocess_waiter.json';mod,state=registry()
if path.exists() and mod.identity(read(path)['pid']) is not None:raise RuntimeError('POSTPROCESS_ALREADY_OWNED')
p=subprocess.Popen([os.sys.executable,'-B',str(ROOT/'postprocess.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=(ROOT/'outputs/postprocess_controller.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
write(path,dict(pid=p.pid,kind='one-shot event continuation, not scheduled task'))
print(p.pid)
