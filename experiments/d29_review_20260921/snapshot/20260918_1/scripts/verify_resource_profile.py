"""Verify the completed controller-only resource update and append its receipt."""
import time,json
import native_runtime as rt
from apply_resource_profile import event
R=rt.RUN
def main():
 folder=R/'reports/resource_reservation_change';c=rt.read(folder/'completed.json');assert c['changed'];rt.process(c['new_controller']['pid'],c['new_controller']['created'])
 old=rt.read(folder/'original_launch_validation.json');new=rt.read(R/'reports/launch_validation.json');assert old['frozen_hashes']==new['frozen_hashes'];assert rt.read(folder/'worker_manifest_hashes.json')=={p.name:rt.sha(p) for p in (R/'reports/launch_by_fold').glob('*.json')}
 before=rt.read(folder/'original_campaign.json');after=rt.read(R/'work/campaign.json');assert all(before[k]==after[k] for k in before if k!='launch_sha256')
 recovery=rt.read(R/'reports/controller_recovery.json');assert recovery['adopted'];status=rt.read(R/'work/controller_status.json');assert status['process']['pid']==c['new_controller']['pid']
 event('RESOURCE_REPROFILE_APPLIED_CONFIRMED',old_peak=c['old_peak'],new_peak=c['new_peak'],controller=c['new_controller'],adopted=[dict(tag=s['tag'],pid=s['pid'],calls=s['calls']) for s in recovery['adopted']],worker_manifests_unchanged=True)
 rt.write(folder/'verified.json',dict(status='PASS_RESOURCE_ONLY_RESTART',time=time.time(),adopted=recovery['adopted'],resources=status['resources'],live=status['live'],bookkeeping_note='The post-restart receipt initially had a duplicate time keyword; controller restart succeeded and this independent receipt repairs only that event.'))
 print('PASS_RESOURCE_ONLY_RESTART',len(status['live']),status['resources'],flush=True)
if __name__=='__main__':main()
