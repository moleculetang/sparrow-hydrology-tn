"""Pinned read-only old kernels. New campaign modules must use fc_ names."""
import sys
from fc_io import TEST

_prior_path = sys.path.copy()
try:
    sys.path.insert(0, str(TEST / '20260907_3/scripts'))
    from tn_reference import load_data as baseline_data, route
    from tn_autograd import RiverN, monthly_sum, boundary_mass
    from baseline_objective import ExtendedObjective
    from fit_models import projected_gradient
    from audit_inputs import HYDRO, SOURCE, TOPO
    from process_info import Process, NoSuchProcess, AccessDenied
    from independent_process import decode as control_decode, replay as control_replay
finally:
    sys.path[:] = _prior_path

# Old FitObjective imports set four threads. The new runtime always overrides.
import torch
torch.set_num_threads(1)
