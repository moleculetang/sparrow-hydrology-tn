"""Read-only event tail for this active conversation; not a scheduled task."""
import time,json
import native_runtime as rt
R=rt.RUN

def main():
    log=R/'work/resource_events.jsonl';position=log.stat().st_size
    state=rt.read(R/'work/controller_status.json');print('MONITOR',state['status'],'audited',state.get('audited'),flush=True)
    while True:
        with log.open('r',encoding='utf-8') as stream:
            stream.seek(position)
            while True:
                start=stream.tell();line=stream.readline()
                if not line or not line.endswith('\n'):position=start;break
                event=json.loads(line);kind=event.get('event')
                if kind in ['DISPATCH','CHILD_EXIT','PAUSE_DISPATCH','RESUME_DISPATCH','CONTROLLER_TERMINAL']:
                    print(time.strftime('%H:%M:%S'),kind,event.get('tag',''),('audit' if event.get('audit') else ''),event.get('exit_code',''),event.get('status',''),flush=True)
        state=rt.read(R/'work/controller_status.json')
        if state['status']!='RUNNING':print('TERMINAL',state['status'],flush=True);return
        if (R/'work/controller_failure.json').exists():print('CONTROLLER_FAILURE',flush=True);raise SystemExit(2)
        time.sleep(2)
if __name__=='__main__':main()
