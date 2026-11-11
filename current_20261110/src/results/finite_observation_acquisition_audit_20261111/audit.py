"""Read-only B0 metadata/coordinate audit and saved source-task provenance."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime, timezone
import csv
import hashlib
import json
import socket
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSV = Path('/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv')
GENERATION = ROOT/'results/claude_hurdle_b0_finite_obs_ens5_20261111'
started = time.monotonic()
assert socket.gethostname() == 'ray-web'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(CSV) == '45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168'
rows = {}
with CSV.open(newline='') as stream:
    for row_id, row in enumerate(csv.DictReader(stream), 1):
        if int(row['BUILDINGID']) == 0:
            rows[row_id] = dict(id=row_id,floor=int(row['FLOOR']),
                xy=(float(row['LONGITUDE']),float(row['LATITUDE'])),
                group=(int(row['USERID']),int(row['PHONEID'])),timestamp=int(row['TIMESTAMP']))
assert len(rows) == 5249

def group_name(group):
    return f'U{group[0]}_P{group[1]}'

def describe(values):
    a = np.asarray(values)
    return dict(minimum=float(a.min()),median=float(np.median(a)),
        p90=float(np.quantile(a,.9,method='inverted_cdf')),maximum=float(a.max()))

def coordinates(selected):
    return sorted({r['xy'] for r in selected})

def coverage(a,b):
    a,b = np.asarray(a),np.asarray(b)
    distances = np.linalg.norm(a[:,None]-b[None],axis=2)
    da,db = distances.min(1),distances.min(0)
    exact = len(set(map(tuple,a)) & set(map(tuple,b)))
    return dict(a_positions=len(a),b_positions=len(b),exact_shared_positions=exact,
        a_to_b_within4m=int((da<=4).sum()),b_to_a_within4m=int((db<=4).sum()),
        a_to_b_coverage=float((da<=4).mean()),b_to_a_coverage=float((db<=4).mean()),
        a_to_b_nearest_m=describe(da),b_to_a_nearest_m=describe(db))

groups = defaultdict(list)
for row in rows.values():
    groups[(row['floor'],row['group'])].append(row)
floor_summaries,temporal = {},{}
for (floor,group), selected in sorted(groups.items()):
    name = group_name(group)
    position_counts = Counter(r['xy'] for r in selected)
    counts = list(position_counts.values())
    floor_summaries.setdefault(str(floor),{})[name] = dict(USERID=group[0],PHONEID=group[1],
        scans=len(selected),positions=len(position_counts),n_histogram=dict(sorted(Counter(counts).items())),n=describe(counts))
    ordered = sorted(selected,key=lambda r:(r['timestamp'],r['id']))
    times = np.asarray([r['timestamp'] for r in ordered],dtype=np.int64)
    gaps = np.diff(times)
    largest = np.argsort(-gaps,kind='stable')[:5]
    utc = lambda t: datetime.fromtimestamp(int(t),timezone.utc).isoformat()
    days = defaultdict(list)
    position_times = defaultdict(set)
    for r in ordered:
        days[utc(r['timestamp'])[:10]].append(r)
        position_times[r['xy']].add(r['timestamp'])
    daily = {day:dict(scans=len(rs),positions=len(coordinates(rs))) for day,rs in days.items()}
    day_pairs = [dict(a=a,b=b,**coverage(coordinates(days[a]),coordinates(days[b]))) for a,b in combinations(days,2)]
    temporal[f'F{floor}_{name}'] = dict(scans=len(selected),distinct_timestamps=len(set(times.tolist())),
        start_epoch=int(times[0]),end_epoch=int(times[-1]),start_UTC=utc(times[0]),end_UTC=utc(times[-1]),
        time_span_seconds=int(times[-1]-times[0]),adjacent_sorted_scan_gaps_seconds=describe(gaps),
        zero_gap_count=int((gaps==0).sum()),original_csv_order_timestamp_decreases=sum(
            a['timestamp']>b['timestamp'] for a,b in zip(selected,selected[1:])),
        largest_five_gaps=[dict(seconds=int(gaps[i]),before_UTC=utc(times[i]),after_UTC=utc(times[i+1]),
            positions_before=len(coordinates(ordered[:i+1])),positions_after=len(coordinates(ordered[i+1:])),
            exact_positions_repeated_across_gap=len(set(coordinates(ordered[:i+1])) & set(coordinates(ordered[i+1:])))) for i in largest],
        UTC_day_counts=daily,day_pair_coverage=day_pairs,
        distinct_timestamps_per_position=describe([len(v) for v in position_times.values()]),
        positions_with_multiple_timestamps=sum(len(v)>1 for v in position_times.values()))

within_floor = {}
for floor in range(4):
    names = sorted(group for f,group in groups if f==floor)
    within_floor[str(floor)] = [dict(a=group_name(a),b=group_name(b),
        **coverage(coordinates(groups[(floor,a)]),coordinates(groups[(floor,b)]))) for a,b in combinations(names,2)]
cross_floor = {}
for group in sorted({g for f,g in groups}):
    floors = sorted(f for f,g in groups if g==group)
    cross_floor[group_name(group)] = dict(floors=floors,by_floor={str(f):dict(
        scans=len(groups[(f,group)]),positions=len(coordinates(groups[(f,group)])),
        fraction_of_floor_positions=len(coordinates(groups[(f,group)]))/len(coordinates([r for r in rows.values() if r['floor']==f]))) for f in floors},
        XY_coverage_between_floors=[dict(a_floor=a,b_floor=b,
            **coverage(coordinates(groups[(a,group)]),coordinates(groups[(b,group)]))) for a,b in combinations(floors,2)])

def composition(ids):
    selected = [rows[int(i)] for i in ids]
    grouped = defaultdict(list)
    for row in selected:
        grouped[row['group']].append(row)
    return dict(scans=len(selected),positions=len(coordinates(selected)),
        distinct_USER_PHONE_groups=len(grouped),
        groups={group_name(group):dict(scans=len(rs),positions=len(coordinates(rs))) for group,rs in sorted(grouped.items())})

complete = json.loads((GENERATION/'complete.json').read_text())
task_inputs,task_audit = [],[]
for record in complete['models']:
    path = GENERATION/record['source_tasks']['path']
    assert sha(path)==record['source_tasks']['sha256']
    tasks = json.loads(path.read_text())['tasks']
    task_inputs.append(dict(seed=record['seed'],path=str(path),sha256=sha(path),
        task_modes=dict(Counter(t['mode'] for t in tasks)),analyzed_Hsingle_composition=record['seed']==1))
    if record['seed'] != 1:
        continue
    for task in tasks:
        if not task['mode'].startswith('Hsingle'):
            continue
        base = set(task['base_S_csv_rows'])
        qids = np.asarray([i for i,r in rows.items() if r['floor']==task['floor'] and i not in base],dtype=np.int64)
        a = np.ascontiguousarray(qids)
        assert len(qids)==task['Q']['count']
        assert hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()==task['Q']['csv_row_ids_sha256']
        task_audit.append(dict(floor=task['floor'],K=task['K'],mode=task['mode'],
            history={floor:composition(value['csv_row_ids']) for floor,value in task['H_rows'].items()},
            base_S=composition(task['base_S_csv_rows']),S=composition(task['S_csv_rows']),Q=composition(qids)))
assert len(task_audit)==24
history_group_counts = [h['distinct_USER_PHONE_groups'] for t in task_audit for h in t['history'].values()]
result = dict(complete=True,created=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,
    runtime=dict(python=sys.executable,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),
        OMP_NUM_THREADS=os.environ['OMP_NUM_THREADS'],GPU_visible=os.environ['CUDA_VISIBLE_DEVICES']),
    scope='B0 metadata and exact coordinates only; no RSS/model/target data analysis',
    definitions=dict(position='Exact raw numeric LONGITUDE/LATITUDE within floor, no rounding',
        group='Recorded USERID/PHONEID pair, not automatically an independent acquisition session',
        coverage='Deduplicated XY, exact overlap and directed nearest distance <=4m; across floors is XY overlap only',
        temporal='Sorted integer TIMESTAMP seconds, UTC date summaries and largest gaps; no threshold/session partition'),
    inputs=dict(csv=dict(path=str(CSV),sha256=sha(CSV)),source_tasks=task_inputs,
        script=dict(path=str(Path(__file__)),sha256=sha(Path(__file__)))),
    B0_scans=len(rows),B0_positions=len({(r['floor'],r['xy']) for r in rows.values()}),
    floors=floor_summaries,within_floor_group_coverage=within_floor,cross_floor_group_coverage=cross_floor,
    temporal_by_floor_group=temporal,seed1_Hsingle_tasks=task_audit,
    seed1_mix_summary=dict(tasks=len(task_audit),history_floor_entries=len(history_group_counts),
        history_group_count_histogram=dict(Counter(history_group_counts)),
        S_group_count_histogram=dict(Counter(t['S']['distinct_USER_PHONE_groups'] for t in task_audit)),
        Q_group_count_histogram=dict(Counter(t['Q']['distinct_USER_PHONE_groups'] for t in task_audit))))
Path(__file__).with_name('audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(seconds=result['seconds'],floors=result['floors'],mix=result['seed1_mix_summary']),indent=2))
