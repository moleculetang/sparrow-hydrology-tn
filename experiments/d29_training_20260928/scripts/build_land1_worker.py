"""Build the finite worker before any gate opens; never dispatch from here."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if any((ROOT/'scripts'/name).exists() for name in ['run_land1_job.py','run_land1_queue.py']):
    raise SystemExit('Maintained LAND1 workers already exist. Do not regenerate over registered execution limits or checkpoint identities; edit and review explicit worker sources before starting paths.')
s=(ROOT/'scripts/run_u_job.py').read_text(encoding='utf-8')
s=s.replace('from d29_training.u_adapter import UTraining',
'''import d29_training.annual_chain as annual_module
from d29_platform import precision_candidate as precise
annual_module.run_land1=precise.run_land1
annual_module.land1_adjoint=precise.land1_adjoint
from d29_training.land1_adapter import Land1Training''')
begin=s.index("    gate=json.loads(")
end=s.index("    job=next",begin)
s=s[:begin]+'''    gate_path=ROOT/'outputs/land1_formal_gate.json'
    if not gate_path.exists():raise RuntimeError('LAND1_GATE_MISSING')
    gate=json.loads(gate_path.read_text(encoding='utf-8'))
    if not gate.get('passed'):raise RuntimeError('LAND1_GATE_NOT_PASSED')
''' + s[end:]
s=s.replace("if job['model']!='U':raise ValueError('U_ONLY_WORKER')","if job['model']!='LAND1':raise ValueError('LAND1_ONLY_WORKER')")
s=s.replace('a=UTraining(job)','a=Land1Training(job)')
s=s.replace("    job=next", "    input_manifest=ROOT/'evidence/runtime_input_identity.json'\n    for path,digest in json.loads(input_manifest.read_text(encoding='utf-8'))['files'].items():\n        if sha(path)!=digest:raise RuntimeError('FROZEN_RUNTIME_INPUT_CHANGED '+path)\n    job=next",1)
s=s.replace('reserve_bytes=4_000_000_000','reserve_bytes=9_000_000_000').replace('reserve_bytes=1_000_000_000','reserve_bytes=4_500_000_000')
s=s.replace("paths=[ROOT/'d29_training/u_adapter.py',ROOT/'d29_training/objective.py',Path(__file__)]", "paths=[ROOT/'d29_training/u_adapter.py',ROOT/'d29_training/land1_adapter.py',ROOT/'d29_training/annual_chain.py',ROOT/'d29_training/objective.py',ROOT/'d29_platform/precision_candidate.py',ROOT/'d29_platform/precision_transfer.py',Path(__file__)]")
(ROOT/'scripts/run_land1_job.py').write_text(s,encoding='utf-8')
q=(ROOT/'scripts/run_u_queue.py').read_text(encoding='utf-8')
q=q.replace("j['model']=='U'","j['model']=='LAND1'").replace('u_queue','land1_queue').replace('u_dispatch','land1_dispatch')
q=q.replace('maximum_concurrent_U_workers\':2','maximum_concurrent_LAND1_workers\':1').replace('len(occupied)<2','len(occupied)<1')
q=q.replace('reserve_bytes=4_000_000_000','reserve_bytes=9_000_000_000').replace('run_u_job.py','run_land1_job.py')
q=q.replace('children={};failed={};paused=False',"gate_path=ROOT/'outputs/land1_formal_gate.json'\nif not gate_path.exists() or not json.loads(gate_path.read_text(encoding='utf-8')).get('passed'):raise RuntimeError('LAND1_FORMAL_GATE_REQUIRED_BEFORE_QUEUE')\nchildren={};failed={};paused=False")
(ROOT/'scripts/run_land1_queue.py').write_text(q,encoding='utf-8')
