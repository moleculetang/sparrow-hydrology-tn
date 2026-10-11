"""Exact old evaluator with result-folder mapping and pre-label contract gate."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from d29_platform.runtime import sha,write_json

def main():
    base=ROOT/'scripts/evaluate_campaign.py'
    gate=json.loads((ROOT/'evidence/formal_gate.json').read_text(encoding='utf8'))
    if gate['identities'].get('scripts/evaluate_campaign.py')!=sha(base):
        raise RuntimeError('SEALED_EVALUATOR_SOURCE_CHANGED')
    original=base.read_text(encoding='utf8')
    replacements=[
        ("def main():\n    import numpy as np,pandas as pd\n",
         "def main():\n    import numpy as np,pandas as pd\n    from approved_epoch_delivery import admitted_folders\n    folder_map=admitted_folders(ROOT)\n    def approved_folder(jid):return folder_map[jid]\n"),
        ("        f=ROOT/'outputs/jobs'/j['id']\n","        f=approved_folder(j['id'])\n"),
        ("base=ROOT/'outputs/jobs'/baseid;candidate=ROOT/'outputs/jobs'/candidateid;",
         "base=approved_folder(baseid);candidate=approved_folder(candidateid);")]
    text=original
    for needle,replacement in replacements:
        if text.count(needle)!=1:raise RuntimeError('ONE_EXACT_EVALUATOR_REPLACEMENT_REQUIRED')
        text=text.replace(needle,replacement)
    reversed_text=text
    for needle,replacement in reversed(replacements):reversed_text=reversed_text.replace(replacement,needle)
    if reversed_text!=original:raise RuntimeError('UNREGISTERED_EVALUATION_METHOD_CHANGE')
    target=ROOT/'analysis/evaluate_approved_campaign.py'
    compile(text,str(target),'exec');target.write_bytes(text.encode('utf8'))
    write_json(ROOT/'evidence/approved_evaluator_build.json',dict(passed=True,
        original_sha256=sha(base),approved_sha256=sha(target),
        policy_sha256=sha(ROOT/'analysis/approved_epoch_delivery.py'),
        unchanged_metrics_scopes_entry_selection_bootstrap=True,
        only_changes=['pre-label approved scientific-epoch gate','immutable folder selection'],
        evaluation_dispatched=False,evaluation_labels_read=False))
    print('passed: exact existing evaluation with approved epoch gate before label reads')

if __name__=='__main__':main()
