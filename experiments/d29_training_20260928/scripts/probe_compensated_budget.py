"""Diagnostic only: audit represented hi-minus-compensation organic stocks.

Does not replace the production kernel or relax its current acceptance gate.
The generated instrumented snapshot records pre-existing compensation values;
it never sets a stock using an observed budget residual.
"""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
import importlib.util
source=ROOT/'d29_platform/land1.py'
text=source.read_text(encoding='utf-8')
text=text.replace('    unmet = np.zeros_like(out)\n','    unmet = np.zeros_like(out)\n    represented_worst = 0.\n    daily_worst = 0.\n    transfer_worst = 0.\n')
text=text.replace('                worst = max(worst, abs(transfer_error))','                worst = max(worst, abs(transfer_error))\n                transfer_worst = max(transfer_worst, abs(transfer_error))')
text=text.replace('            P, SA, SP, N, L = before[u]\n            ka, kp', '            old_correction = organic_correction[u].copy()\n            P, SA, SP, N, L = before[u]\n            ka, kp',1)
text=text.replace('            worst = max(worst, abs(balance))\n    return flux, states, worst, -1, -1, 0.0, organic_correction, unmet',
'''            worst = max(worst, abs(balance))
            daily_worst = max(daily_worst, abs(balance))
            represented = _accurate_sum(np.concatenate((states[newi,u],-before[u],-source[t,u],flux[t,u],old_correction,-organic_correction[u])))
            represented_worst = max(represented_worst,abs(represented))
    return flux, states, worst, -1, -1, 0.0, organic_correction, unmet, represented_worst, daily_worst, transfer_worst''')
text=text.replace('return flux, states, worst, t, u, -plant_after, organic_correction, unmet','return flux, states, worst, t, u, -plant_after, organic_correction, unmet, represented_worst, daily_worst, transfer_worst')
text=text.replace('flux, states, err, day, unit, short, correction, unmet = _forward','flux, states, err, day, unit, short, correction, unmet, represented_worst, daily_worst, transfer_worst = _forward')
text=text.replace("    inp['realized_outflows']=inp['outflows']-unmet", "    inp['realized_outflows']=inp['outflows']-unmet\n    inp['budget_probe']={'represented_daily_max':represented_worst,'rounded_daily_max':daily_worst,'transfer_max':transfer_worst}")
snapshot=ROOT/'work/precision_probe_kernel.py';snapshot.write_text(text,encoding='utf-8')
spec=importlib.util.spec_from_file_location('precision_probe_kernel',snapshot)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
# Reuse the diagnostic scenario construction, never mutate the producer.
driver=(ROOT/'scripts/diagnose_land1_precision.py').read_text(encoding='utf-8')
driver=driver.replace('from d29_platform.land1 import run_land1,hazard_to_probability','from precision_probe_kernel import run_land1,hazard_to_probability')
driver=driver.replace("kernel_max=b.max_local_balance_kg,", "kernel_max=b.max_local_balance_kg,budget_probe=b._inputs['budget_probe'],")
driver=driver.replace("outputs/land1_precision_diagnosis.json","outputs/compensated_budget_probe.json")
write_json(ROOT/'evidence/compensated_budget_probe_identity.json',{'parent_sha256':sha(source),'instrumented_sha256':sha(snapshot),'purpose':'Separate rounded high-part residual from represented-state residual; no formal acceptance'})
exec(compile(driver,str(ROOT/'scripts/diagnose_land1_precision.py'),'exec'))
