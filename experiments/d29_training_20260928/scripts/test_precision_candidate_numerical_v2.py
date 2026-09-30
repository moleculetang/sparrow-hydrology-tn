"""Reuse the independent reference tests against the isolated candidate API."""
import sys,unittest,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
from d29_platform import precision_candidate as candidate
from d29_platform.precision_tags_candidate import propagate_source_labels
if '--mixture' in sys.argv:
    from d29_platform.mixture_tags_candidate import propagate_source_labels
spec=importlib.util.spec_from_file_location('precision_reference_tests',ROOT/'tests/test_land1.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
for name in ['run_land1','land1_adjoint','PlantBudgetInfeasible','MissingLand1Input','PROBABILITY_NAMES','hazard_to_probability']:
    setattr(m,name,getattr(candidate,name))
m.propagate_source_labels=propagate_source_labels
suite=unittest.defaultTestLoader.loadTestsFromModule(m)
import d29_training.annual_chain as annual
annual.run_land1=candidate.run_land1;annual.land1_adjoint=candidate.land1_adjoint
annual_spec=importlib.util.spec_from_file_location('precision_annual_tests',ROOT/'tests/test_annual_chain.py')
annual_tests=importlib.util.module_from_spec(annual_spec);sys.modules[annual_spec.name]=annual_tests;annual_spec.loader.exec_module(annual_tests)
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(annual_tests))
result=unittest.TextTestRunner(verbosity=2).run(suite)
if '--mixture' in sys.argv:
    write_json(ROOT/'outputs/precision_candidate_test_receipt_numerical_v2.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'identities':{str(p):sha(p) for p in [ROOT/'d29_platform/precision_candidate.py',ROOT/'d29_platform/precision_transfer.py',ROOT/'d29_platform/mixture_tags_candidate.py',ROOT/'d29_training/annual_chain.py']},'scope':'independent small kernel, multistep source/state derivatives, year-boundary initial-state and external-source gradients; full-scale coordinate checks separate'})
raise SystemExit(0 if result.wasSuccessful() else 1)

