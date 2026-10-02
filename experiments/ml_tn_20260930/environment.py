import platform,subprocess,importlib,sys
from mltn.common import ROOT,write
def main():
    versions={}
    for n in ['numpy','pandas','scipy','sklearn','torch','xgboost','lightgbm','catboost']:
        m=importlib.import_module(n);versions[n]=m.__version__
    import torch
    write(ROOT/'evidence'/f'environment_{platform.system()}.json',dict(host=platform.node(),python=sys.version,executable=sys.executable,packages=versions,cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),gpu_available=torch.cuda.is_available(),TF32=False,amp=False))
if __name__=='__main__':main()
