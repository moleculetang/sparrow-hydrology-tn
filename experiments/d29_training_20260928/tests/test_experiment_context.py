import json
import tempfile
import unittest
from pathlib import Path

from d29_platform.runtime import sha
from d29_training.experiment_context import ExperimentContext


class ContextTests(unittest.TestCase):
    def test_same_id_old_checkpoint_cannot_be_used_as_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config').mkdir()
            jobs=[{'id':'LAND1_F23_T0_s0','model':'LAND1'}]
            (root/'config/jobs.json').write_text(json.dumps(jobs),encoding='utf-8')
            (root/'config/experiment_context.json').write_text(json.dumps({
                'root':str(root),'experiment_id':'20260928_1_review_v1','namespace':'jobs',
                'hashes':{'config/jobs.json':sha(root/'config/jobs.json')}}),encoding='utf-8')
            old=root/'outputs/jobs_numerical_v2/LAND1_F23_T0_s0';old.mkdir(parents=True)
            (old/'worker.json').write_text(json.dumps({'experiment_id':'old'}),encoding='utf-8')
            context=ExperimentContext.load(root)
            self.assertEqual(context.folder(jobs[0]['id']),root/'outputs/jobs/LAND1_F23_T0_s0')
            with self.assertRaisesRegex(RuntimeError,'CURRENT_NAMESPACE_WORKER_MISSING'):
                context.check_worker(jobs[0]['id'])
            fresh=context.folder(jobs[0]['id']);fresh.mkdir(parents=True)
            (fresh/'worker.json').write_text(json.dumps({'experiment_id':'wrong'}),encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'WORKER_EXPERIMENT_MISMATCH'):
                context.check_worker(jobs[0]['id'])
            (fresh/'worker.json').write_text(json.dumps({'experiment_id':context.experiment_id}),encoding='utf-8')
            self.assertEqual(context.check_worker(jobs[0]['id'])['experiment_id'],context.experiment_id)
            (root/'config/jobs.json').write_text('[]',encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'EXPERIMENT_IDENTITY_CHANGED'):
                ExperimentContext.load(root)


if __name__=='__main__':unittest.main()
