"""Wait for the preceding verified return, then run two existing local paths serially."""
import time
from mltn.common import ROOT,read
from run_single_external import main

preceding='joint_S23_GraphTCN_c0_s1731_B191'
marker=ROOT/'outputs/stream_returns'/f'{preceding}.json'
while not marker.exists():time.sleep(5)
assert read(marker)['remote_import_verified'] and read(marker)['producer_exit_verified']
for jid in ['joint_S23_GraphTCN_c0_s1731_B56','joint_S23_GraphTCN_c0_s1731_B113']:
    main(jid)
