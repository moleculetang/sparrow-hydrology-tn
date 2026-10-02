"""Per-process peaks; framework GPU allocator scope is explicitly labeled."""
import os
def peaks():
    import torch
    if os.name=='posix':
        import resource
        rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    else:
        import ctypes
        class PM(ctypes.Structure):_fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['peak','working','ppq','pq','pnpq','npq','page','peakpage']]
        m=PM();m.cb=ctypes.sizeof(m);rss=int(m.peak) if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(m),m.cb) else None
    from threadpoolctl import threadpool_info
    return dict(process_peak_working_bytes=rss,torch_peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else None,gpu_peak_scope='PyTorch allocator only; external tree libraries not measured by this allocator',actual_library_threadpools=threadpool_info(),cpu_affinity=sorted(os.sched_getaffinity(0)) if os.name=='posix' else None)
