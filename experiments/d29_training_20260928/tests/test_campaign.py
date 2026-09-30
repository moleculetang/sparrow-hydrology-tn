import unittest
from d29_training.campaign import select_campaign, year_role


def fixtures():
    jobs = [dict(id=f'T{s}_s{e}', model='U', fold='F23', strategy=f'T{s}',
                 space_block=None, entry=e, train_end=2022) for s in [0,2] for e in [0,1]]
    records = {j['id']: dict(status='solver_stopped_not_sufficient', objective=2-j['entry'],
                            objective_passed=True, physical_passed=True) for j in jobs}
    return jobs, records


class CampaignTests(unittest.TestCase):
    def test_training_selection_and_full_accounting(self):
        jobs, records = fixtures()
        assert select_campaign(jobs, records, {})['selected_jobs'] == ['T0_s1', 'T2_s1']
        with self.assertRaises(ValueError): select_campaign(jobs, {}, {})
        records['T0_s0']['status'] = 'resource_checkpoint'
        with self.assertRaises(ValueError): select_campaign(jobs, records, {})
    
    
    def test_audit_and_baseline_required(self):
        jobs, records = fixtures()
        records['T0_s1']['physical_passed'] = False
        with self.assertRaises(ValueError): select_campaign(jobs, records, {})
        records = {k:v for k,v in records.items() if k.startswith('T2')}
        with self.assertRaises(ValueError): select_campaign(jobs, records, {'T0_s0':'failed', 'T0_s1':'failed'})
    
    
    def test_t0_early_history_is_not_training(self):
        job=fixtures()[0][0]
        assert year_role(job,2018)=='historical_backcast_not_training'
        assert year_role(job,2022)=='training_reconstruction'
        assert year_role(job,2023)=='retrospective_time_evaluation'
    
