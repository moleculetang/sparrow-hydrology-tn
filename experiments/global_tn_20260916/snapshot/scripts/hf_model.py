"""Daily output selections use the unchanged full-history physical adjoint."""
from temporal_model import *
class HFEndpoints(TemporalEndpoints):
 def daily_metadata(self,meta):
  if 'day_index' not in meta:return super().daily_metadata(meta)
  forbidden=set(meta)-set(META_FIELDS)
  if forbidden:raise ValueError('NONWHITELIST_PREDICTION_METADATA '+str(forbidden))
  c,record,weights=super().daily_metadata(meta.drop(columns='day_index'))
  requested=meta.day_index.to_numpy(int)[record.numpy()]
  selected=(requested<0)|(c['ti'].numpy()==requested)
  counts=np.bincount(record.numpy()[selected],minlength=len(meta))
  if np.any(counts==0) or np.any(counts[meta.day_index.to_numpy(int)>=0]!=1):raise ValueError('INVALID_DAILY_INDEX')
  mask=torch.tensor(selected)
  return {k:v[mask] if isinstance(v,torch.Tensor) else v for k,v in c.items()},record[mask],weights[mask]
