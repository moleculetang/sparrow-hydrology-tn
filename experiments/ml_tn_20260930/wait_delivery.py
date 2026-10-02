"""Silent single SSH session; completion event only, no periodic remote dispatch."""
import time,json
from mltn.common import ROOT,read
from mltn.resources import registry
while True:
    receipt=ROOT/'delivery_archive_receipt.json'
    if receipt.exists():print(json.dumps(read(receipt)),flush=True);break
    for marker in ['postprocess_waiter.json','remaining_owner.json']:
        path=ROOT/'outputs'/marker
        if path.exists():
            mod,state=registry();pid=read(path)['pid']
            completion=ROOT/'delivery_archive_receipt.json' if marker=='postprocess_waiter.json' else ROOT/'outputs/all_training_done.json'
            if mod is not None and mod.identity(pid) is None and not completion.exists():raise RuntimeError('EVENT_CONTINUATION_FAILED '+marker+'; inspect retained controller log')
    time.sleep(15)
