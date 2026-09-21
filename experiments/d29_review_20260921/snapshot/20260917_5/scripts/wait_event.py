"""Wait on controller exit/attention, no iterative status-file polling."""
import ctypes,hashlib,time
import native_runtime as rt
name='Local\\SPARROW_attention_'+hashlib.sha256(str(rt.RUN).encode()).hexdigest()[:16]
rt.k32.OpenEventW.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_wchar_p];rt.k32.OpenEventW.restype=ctypes.c_void_p
event=rt.k32.OpenEventW(0x100000|2,False,name)
status=rt.read(rt.RUN/'work/controller_status.json');proc=status['process'];rt.process(proc['pid'],proc['created'])
handle=rt.k32.OpenProcess(0x100000|0x1000,False,proc['pid'])
handles=(ctypes.c_void_p*2)(handle,event)
rt.k32.WaitForMultipleObjects.argtypes=[ctypes.c_ulong,ctypes.POINTER(ctypes.c_void_p),ctypes.c_int,ctypes.c_ulong]
rt.k32.ResetEvent.argtypes=[ctypes.c_void_p]
start=time.monotonic()
while time.monotonic()-start<3600:
 result=rt.k32.WaitForMultipleObjects(2,handles,False,60000)
 if result==0:print('CONTROLLER_EXIT',flush=True);break
 if result==1:rt.k32.ResetEvent(event);print('ATTENTION',flush=True);break
 if result!=258:raise RuntimeError(('WAIT_FAILED',result))
else:print('HOURLY_CHECK_DUE',flush=True)
