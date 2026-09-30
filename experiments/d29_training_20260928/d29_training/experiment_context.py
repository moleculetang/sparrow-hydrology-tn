"""Immutable path and input identity for the 2026-09-28 review campaign."""
from dataclasses import dataclass
import json
from pathlib import Path

from d29_platform.runtime import ROOT, sha


@dataclass(frozen=True)
class ExperimentContext:
    root: Path
    experiment_id: str
    namespace: str
    hashes: dict

    @classmethod
    def load(cls, root=ROOT):
        root=Path(root).resolve()
        record=json.loads((root/'config/experiment_context.json').read_text(encoding='utf-8'))
        if record['namespace']!='jobs' or record['experiment_id']!='20260928_1_review_v1':
            raise RuntimeError('UNREGISTERED_EXPERIMENT_CONTEXT')
        if Path(record['root']).resolve()!=root:
            raise RuntimeError('EXPERIMENT_ROOT_CHANGED')
        for relative,digest in record['hashes'].items():
            path=(root/relative).resolve()
            if not path.is_relative_to(root) or sha(path)!=digest:
                raise RuntimeError('EXPERIMENT_IDENTITY_CHANGED '+relative)
        return cls(root,record['experiment_id'],record['namespace'],record['hashes'])

    def job(self, job_id):
        jobs=json.loads((self.root/'config/jobs.json').read_text(encoding='utf-8'))
        matches=[j for j in jobs if j['id']==job_id]
        if len(matches)!=1:raise RuntimeError('UNREGISTERED_JOB '+job_id)
        return dict(matches[0],output_namespace=self.namespace,experiment_id=self.experiment_id)

    def folder(self, job_id):
        self.job(job_id)
        return self.root/'outputs'/self.namespace/job_id

    def check_worker(self, job_id):
        folder=self.folder(job_id)
        path=folder/'worker.json'
        if not path.exists():raise RuntimeError('CURRENT_NAMESPACE_WORKER_MISSING '+job_id)
        receipt=json.loads(path.read_text(encoding='utf-8'))
        if receipt.get('experiment_id')!=self.experiment_id:
            raise RuntimeError('WORKER_EXPERIMENT_MISMATCH '+job_id)
        return receipt
