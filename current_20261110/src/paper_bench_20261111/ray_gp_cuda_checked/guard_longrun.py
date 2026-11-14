"""Run the verified executor while bounding host memory and surfacing failures."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

WORK = Path(__file__).resolve().parent
BASE = Path('/home/panyushuo/projects/panyushuo/paper_bench_wt/current_20261110')
OUTPUT = BASE / 'results/paper_tampere_TIE1_ext_cuda_20261114'
MEMORY_EVENTS = Path('/sys/fs/cgroup/user.slice/user-1003.slice/memory.events')


def write_status(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def changed_failures(directory, initial_times):
    return [str(path) for path in directory.rglob('*.json')
        if path.stat().st_mtime_ns != initial_times.get(path)
        and json.loads(path.read_text()).get('status') != 'completed']


def main():
    before = MEMORY_EVENTS.read_text()
    started = time.time()
    event_directory = OUTPUT / 'events/refi_GP'
    initial_times = {path: path.stat().st_mtime_ns for path in event_directory.rglob('*.json')}
    log = open(WORK / 'launcher.log', 'w')
    process = subprocess.Popen(['bash', str(WORK / 'run.sh')],
        start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
    write_status(WORK / 'guard_start.json', dict(pid=process.pid,
        start_epoch=started, memory_events=before))
    peak = 0
    minimum_available = 2**63-1
    reason = 'running'
    last_event_check = 0
    failures = []
    while process.poll() is None:
        rows = subprocess.check_output(['ps', '-eo', 'pgid=,rss='], text=True).splitlines()
        resident = sum(int(values[1])*1024 for row in rows
            if len(values := row.split()) == 2 and int(values[0]) == process.pid)
        available = int(next(line.split()[1] for line in
            Path('/proc/meminfo').read_text().splitlines()
            if line.startswith('MemAvailable:')))*1024
        peak = max(peak, resident)
        minimum_available = min(minimum_available, available)
        if resident > 18*1024**3 or available < 8*1024**3:
            reason = 'memory_limit'
            stop(process)
            break
        if time.monotonic()-last_event_check >= 30:
            last_event_check = time.monotonic()
            failures = changed_failures(event_directory, initial_times)
            if failures:
                reason = 'task_failed'
                stop(process)
                break
            write_status(WORK/'heartbeat.json', dict(pid=process.pid,
                time=time.time(), resident_bytes=resident, available_bytes=available,
                peak_group_rss_bytes=peak))
        time.sleep(3)
    status = process.wait()
    log.close()
    records = [json.loads(path.read_text()) for path in (OUTPUT/'events/refi_GP').rglob('*.json')]
    completed = sum(record.get('status') == 'completed' for record in records)
    expected = len(json.loads((OUTPUT/'task_inventory.json').read_text())['ready'])
    export_status = None
    if reason == 'running':
        reason = 'completed' if status == 0 and completed == expected else 'incomplete_or_failed'
    if reason == 'completed':
        with open(WORK/'export_gp.log', 'w') as export_log:
            export_status = subprocess.call(['bash', str(WORK/'run.sh'), '--export-gp'],
                cwd=BASE/'src', stdout=export_log, stderr=subprocess.STDOUT)
        if export_status != 0:
            reason = 'export_failed'
    write_status(WORK/'guard_result.json', dict(pid=process.pid,
        start_epoch=started, end_epoch=time.time(), returncode=status, reason=reason,
        completed_tasks=completed, expected_tasks=expected, failed_paths=failures,
        export_returncode=export_status, peak_group_rss_bytes=peak,
        min_available_bytes=minimum_available, memory_events_before=before,
        memory_events_after=MEMORY_EVENTS.read_text()))
    print(reason, completed, expected, flush=True)
    return 0 if reason == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
