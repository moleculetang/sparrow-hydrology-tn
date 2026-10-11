"""Known compensated cancellation and changed-inventory controls."""
import json
from pathlib import Path
import sys
import numpy as np
from bnf_endpoint_readback import grouped_errors


def main():
    original=np.zeros((230,13,5,5));original_corr=np.zeros((230,13,5,4))
    original[0,5,3,1]=1e8;original_corr[0,5,3,0]=1e-9
    original[0,5,4,2]=3e8
    original[0,5,4,1]=100.
    new=np.zeros_like(original);corr=np.zeros_like(original_corr)
    new[0,5,1,1]=4e7;new[0,5,2,1]=6e7;corr[0,5,1,0]=1e-9
    new[0,5,0,2]=3e8
    new[0,5,0,1]=100.
    check=grouped_errors(new,corr,original,original_corr)
    assert check['passed'] and max(check['max_abs_error_kg'].values())==0.
    wrong=new.copy();wrong[0,5,2,1]+=.1;wrong[0,5,0,1]-=.1
    assert not grouped_errors(wrong,corr,original,original_corr)['passed']
    wrong=corr.copy();wrong[0,5,1,0]=.1
    assert not grouped_errors(new,wrong,original,original_corr)['passed']
    refused=[]
    for values in ((new[:1],corr,original,original_corr),(new,corr,np.full_like(original,np.nan),original_corr)):
        try:grouped_errors(*values)
        except ValueError:refused.append(True)
        else:raise AssertionError('BAD_BNF_SUPPORT_ACCEPTED')
    result=dict(passed=True,known_compensated_equivalence=True,changed_BNF_inventory_rejected=True,
        changed_compensation_rejected=True,incomplete_and_nonfinite_rejected=len(refused)==2,
        actual44_BNF_readback_completed=False,evaluation_labels_read=False)
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    from d29_platform.runtime import sha,write_json
    result['implementation_sha256']={p:sha(root/p) for p in
        ('analysis/bnf_endpoint_readback.py','analysis/test_bnf_endpoint_readback.py')}
    write_json(root/'outputs/bnf_partition_software_controls/endpoint_controls.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
