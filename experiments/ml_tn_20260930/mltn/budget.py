import datetime
from mltn.common import ROOT,read,write

def guard(folder):
    deadline=datetime.datetime.fromisoformat(read(ROOT/'study.json')['deadline_utc'].replace('Z','+00:00'))
    if datetime.datetime.now(datetime.timezone.utc)>=deadline:
        write(folder/'budget_stop.json',dict(status='budget_checkpoint',deadline_utc=deadline.isoformat(),checkpoint='resume.pt',scientific_completion=False))
        raise RuntimeError('ABSOLUTE_STUDY_BUDGET_REACHED_CHECKPOINT_RETAINED')
