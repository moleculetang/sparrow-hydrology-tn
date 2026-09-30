import unittest,copy
from d29_training.derivative_gate import refinement_passes


class DerivativeGateTests(unittest.TestCase):
    def fixture(self):
        scan=dict(identity={'kernel':'abc'},objective=10.,analytic_gradient=[2.])
        d=dict(identity=scan['identity'],coordinate=0,base_value=10.,analytic=2.,checks=[])
        for h in [1e-6,3e-7,1e-7]:
            d['checks'].append(dict(step=h,plus=dict(value=10+2*h,branch_switch_count=0,switches=[]),minus=dict(value=10-2*h,branch_switch_count=0,switches=[])))
        return scan,d

    def test_adjacent_fine_grid_required(self):
        s,d=self.fixture();self.assertTrue(refinement_passes(s,0,d))
        bad=copy.deepcopy(d);bad['checks']=bad['checks'][-1:]
        self.assertFalse(refinement_passes(s,0,bad))
        bad=copy.deepcopy(d);bad['checks'][0]['plus']['value']+=.001;bad['checks'][2]['plus']['value']+=.001
        self.assertFalse(refinement_passes(s,0,bad))

    def test_branch_and_identity_changes_rejected(self):
        s,d=self.fixture();d['checks'][1]['plus']['branch_switch_count']=1
        self.assertFalse(refinement_passes(s,0,d))
        s,d=self.fixture();d['identity']={'kernel':'changed'}
        self.assertFalse(refinement_passes(s,0,d))
