"""One complete forward/backward benchmark for Stage-3 resource planning."""

import json
import ctypes
from ctypes import wintypes
import sys
import time
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reservoir_tn_core import ReservoirTNModel  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
started = time.perf_counter()
model = ReservoirTNModel()
init_seconds = time.perf_counter() - started
parameters = pd.read_parquet(
    r"E:\SPARROW\5_Test\20260824_46\outputs\by_candidate\mineral_lifetime_parameters.parquet"
)
row = parameters.loc[parameters.fold_id.eq("T1")].iloc[0]
model.set_fixed_v_f(float(row["v_f"]))
physical = torch.tensor([row[name] for name in model.names()], requires_grad=True)
obs = pd.read_parquet(r"E:\SPARROW\5_Test\20260824_18\outputs\tn_observations_audited.parquet")
obs = obs.loc[obs.formal_river_channel].copy()
positions = pd.read_parquet(
    r"E:\SPARROW\5_Test\20260824_12\outputs\tn_observations_primary_2016_2024.parquet"
)[["station_key", "reach_id", "downstream_fraction_on_reach"]].drop_duplicates(["station_key", "reach_id"])
obs = obs.merge(positions, on=["station_key", "reach_id"], validate="many_to_one")
train = obs.loc[obs.year.eq(2021)].copy()
started = time.perf_counter()
_, prediction = model.evaluate(train, physical, 2021, 2021)
target = torch.log1p(torch.tensor(train.tn_mg_l.to_numpy(float)))
loss = torch.mean((prediction - target).square())
forward_seconds = time.perf_counter() - started
started = time.perf_counter()
loss.backward()
backward_seconds = time.perf_counter() - started
class Counters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]
counters = Counters(); counters.cb = ctypes.sizeof(counters)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True); psapi = ctypes.WinDLL("psapi", use_last_error=True)
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
print(json.dumps({
    "init_seconds": init_seconds,
    "forward_seconds": forward_seconds,
    "backward_seconds": backward_seconds,
    "rss_gib": counters.WorkingSetSize / 2**30,
    "peak_rss_gib": counters.PeakWorkingSetSize / 2**30,
    "loss": float(loss.detach()),
    "gradient_finite": bool(torch.isfinite(physical.grad).all()),
    "gradient": physical.grad.tolist(),
}, indent=2))
