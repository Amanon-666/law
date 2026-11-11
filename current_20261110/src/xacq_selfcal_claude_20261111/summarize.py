"""Summarize sealed outputs only; never refit or rerun predictions."""
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
def read(p): return json.loads(p.read_text())
def pair(v): return f"{v['mean']:.6f} / {v.get('p90', v.get('position_mean_p90')):.6f}"
lines=['## 完成结果（米）','', 'UJI为原层级位置均值及其加权p90；IPIN为原正式口径已覆盖GT点加权均值及逆经验CDF p90，K不混合。','']
u=read(R/'uji_summary.json')
lines+=['|UJI条件|原A mean / p90|完整自校准 mean / p90|','|---|---:|---:|']
for key in ['overall','1','3','5']:
    a=u['baseline']['overall'] if key=='overall' else u['baseline']['by_K'][key]
    b=u['all']['overall'] if key=='overall' else u['all']['by_K'][key]
    lines.append(f'|{key}|{pair(a)}|{pair(b)}|')
lines+=['',f"P1：Δ={u['P1']['delta']:.6f}米；全部β在区间内={u['P1']['beta_in_range']}。",'', '|UJI建筑，目标层间分布|α min/median/max|β min/median/max|c min/median/max|b min/median/max|','|---|---:|---:|---:|---:|']
uc=read(R/'uji_calibrations.json')
for building in ('B1','B2'):
    cells=[]
    for param in ('alpha','beta','c','b'):
        vals=[v[param] for k,v in uc.items() if k.startswith(building)]
        cells.append('/'.join(f'{v:.6f}' for v in (min(vals),statistics.median(vals),max(vals))))
    lines.append('|'+building+'|'+'|'.join(cells)+'|')
lines+=['','同一目标层的H在六个方向/K事件中完全一致，因此固定seed校准只计算一次并等价复用，未使用事件S。', '', '|IPIN层|α|β|c|b/dB|','|---|---:|---:|---:|---:|']
ic=read(R/'ipin_calibrations.json')
for f,c in ic.items():
    lines.append('|F'+f+'|'+'|'.join(f'{c[k]:.6f}' for k in ('alpha','beta','c','b'))+'|')
scores={m:read(R/'ipin'/m/'scores.json') for m in ('baseline','all','detection','intensity')}
for k,expected in zip(('1','3','5'),(37.2106,37.1091,36.0987)):
    assert abs(scores['baseline']['by_K'][k]['overall']['methods']['hurdle']['mean']-expected)<.00005
summary={m:{k:v for k,v in s.items() if k!='points'} for m,s in scores.items()}
(R/'ipin_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines+=['','|IPIN K|原A|完整校准|仅检出(a)|仅强度(b)|原GP（继承）|','|---|---:|---:|---:|---:|---:|']
for k in ('1','3','5'):
    vals=[pair(scores[m]['by_K'][k]['overall']['methods']['hurdle']) for m in scores]
    vals.append(pair(scores['baseline']['by_K'][k]['overall']['methods']['GP']))
    lines.append('|'+k+'|'+'|'.join(vals)+'|')
lines+=['','|IPIN K/层|原A|完整校准|仅检出|仅强度|','|---|---:|---:|---:|---:|']
for k in ('1','3','5'):
    for f in map(str,range(2,7)):
        vals=[scores[m]['by_K'][k]['by_floor'][f]['methods']['hurdle']['mean'] for m in scores]
        lines.append('|'+k+'/F'+f+'|'+'|'.join(f'{v:.6f}' for v in vals)+'|')
lines+=['','|IPIN K|原固定half mean/p90|自校准固定half mean/p90|','|---|---:|---:|']
for k in ('1','3','5'):
    lines.append('|'+k+'|'+pair(scores['baseline']['by_K'][k]['overall']['methods']['probability_half'])+'|'+pair(scores['all']['by_K'][k]['overall']['methods']['probability_half'])+'|')
lines+=['','参数方向与预定性能门槛分开判断。上述结果不构成新独立测试，不自动替换采用版本；所有经验S条目均未改。']
(R/'result_tables.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
