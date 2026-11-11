"""Render completed server diagnostic; no local experiment."""
import json, sys, os
from pathlib import Path
P=Path(__file__).resolve().parent
r=json.loads((P/'result.json').read_text())
lines=['# IPIN F2 完整自校准后诊断（2026-11-11）','',
'本单由 Cage Claude Code 请求。IPIN 已暴露，全部为事后诊断，不构成方法升级或独立测试。复用 e30afe5 的完整自校准实现与 F2 固定参数，ff0e83d 的证据分解口径；fetch 后起点 253b7aa。只新增探索目录文件，没有训练、模型加载、场重建、参数拟合或数据/模型下载。CPU 两线程、nice 10、无 GPU。',
'','## 口径与输入','',
'每档 K 使用原已评分的全部 22 个 F2 查询点，保持原日志、GT 行、预测 tick 与已到达 WiFi receipt 映射。每个完整校准后验均值逐点与 e30afe5 保存预测核对（绝对容差 1e-8 m）。原 GP 和同候选 GP 均直接读取该次冻结预测。真值仅用于距离、诊断、评分和明确标注的 oracle 限制，不参与校准或原预测。',
'','F2 凸包由全部 480 条训练扫描、389 个不同位置构造，采用原公共米制坐标，边界容差 1e-8 m；距离为到凸区域的欧氏距离，内部记零。S 是原嵌套 K 个位置。历史候选是 F3–F6 缓存位置的去重并集，一个候选可能对应多层，全部列出；经验 S 候选另标 S。来源表示坐标来源，不代表某个历史层单独生成该候选的场。',
'','场仅定义在冻结离散候选上。“真值处 p”统一指真值最近候选的 p，报告其实际距离；最近位置有并列时，在该阶段选分数最高者，与 ff0e83d 相同。不插值或重新推断任意真值坐标上的场。校准前后各自选取 MAP 和并列最近者，索引随行保存。',
'','Δ 均为真值最近候选减 MAP；负值支持错误 MAP。已检出项保留 p、Gaussian 强度、污染项、五个共享偏移的 logsumexp 积分；未检出项为 log((1−EPS)(1−p)+EPS×0.5) 的和；两项均乘 128/AP数。每点两项之和与原后验 log 质量差核对。',
'','已检出 AP 因共享偏移不独立可加：采用从 MAP 到最近真值候选的逐 AP log-likelihood 线性路径积分（Aumann–Shapley），对 offset softmax 权重积分后乘各 AP 变化，逐点加总严格还原已检出总分差（误差<1e-8）。这是归因约定，不是新读法。未检出 AP 贡献直接相减。Top20 按 22 点平均贡献的绝对值排序并保留正负号；未出现的项贡献为零。提供完整逐点/AP数组，可复查其它排序。',
'','AP 表 F2/F3/F4/F5/F6 为各层全部训练 receipts 的检出率（非位置均值）；H3–H6 为实际采用历史缓存的检出率，避免把全部训练和实际 H 混同。S 向量按原抽样顺序，每位 1/0 表示该扫描检出/未检出。p 两列是 22 点平均，逐点原数在 AP_raw_K*.npz。',
'','候选限制保持校准后原分数不变，只在允许集合内重新归一化；并列报告均值和 MAP 误差。S30/S60 是任一 S 位置半径并集；oracle 凸包利用全部目标训练位置，只是额外信息诊断参照，不能称可部署方法或严格理论误差上限。',
'',f"固定校准参数：α={r['calibration']['alpha']:.12g}，β={r['calibration']['beta']:.12g}，c={r['calibration']['c']:.12g}，b={r['calibration']['b']:.12g} dB。",'']
def table(headers,rows):
    lines.append('|'+ '|'.join(headers)+'|'); lines.append('|'+ '|'.join(['---']*len(headers))+'|')
    for row in rows: lines.append('|'+ '|'.join(str(x) for x in row)+'|')
    lines.append('')
def f(v): return f'{v:.6f}'
def xy(v): return ', '.join(f(x) for x in v)
lines+=['## 1. 位置与区域','']
table(['K','候选总数/凸包内','真值内','MAP内','均值内','GP内','同候选GP内','S内'],[[s['K'],f"{s['candidates']}/{s['candidate_inside']}",*[f"{s['inside_counts'][n]}/22" for n in ('truth','MAP','mean','GP','GP_candidates')],'/'.join(str(int(v)) for v in s['S_inside'])] for s in r['summaries']])
table(['K','原均值误差','MAP误差','GP误差','同候选GP误差','均值到凸包','MAP到凸包'],[[s['K'],*[f(s['means'][n]) for n in ('mean_error','MAP_error','GP_error','GP_candidates_error','mean_hull_distance','MAP_hull_distance')]] for s in r['summaries']])
table(['K','S序号','S坐标 m','内/外','到凸包 m'],[[s['K'],i,xy(p),int(s['S_inside'][i]),f(s['S_hull_distance'][i])] for s in r['summaries'] for i,p in enumerate(s['S_xy'])])
lines+=['## 2. 证据分解','']
table(['K','阶段','已检出Δ均值','未检出Δ均值','总Δ均值','最近候选平均秩','已检出正确但被未检出推翻'],[[s['K'],phase,*[f(s['decomposition'][phase][key]) for key in ('detected_delta','missing_delta','total_delta','nearest_rank')],f"{s['missing_overrides'][phase]}/22"] for s in r['summaries'] for phase in ('before','after')])
lines+=['## 3. 主导 AP','']
for k in (1,3,5):
    for term in ('missing','detected'):
        lines += [f'### K{k} {term} Top20','']
        table(['序','AP','Δ均值','F2','F3','F4','F5','F6','H3','H4','H5','H6','S检出','p真值最近','pMAP','Q检出/22'],[[a['rank'],a['ap'],f(a['delta']),*[f(a['training_rates'][str(n)]) for n in range(2,7)],*[f(a['H_rates'][str(n)]) for n in range(3,7)],''.join(str(int(v)) for v in a['S_detected']),f(a['p_truth']),f(a['p_MAP']),a['query_detected_count']] for a in r['top_APs'] if a['K']==k and a['term']==term])
lines+=['## 4. 只读反事实','']
table(['K','允许区域','保留候选数','后验均值误差 m','MAP误差 m'],[[s['K'],name,s['candidate_mask_counts'][name],f(s['means'][name+'_restricted_mean_error']),f(s['means'][name+'_restricted_MAP_error'])] for s in r['summaries'] for name in ('all','S30','S60','oracle_hull')])
lines+=['## 逐点原数','', '以下距离及误差单位均为米；显示六位小数，result.json 保留浮点原数。坐标为 (x,y)。内/距为是否在凸包内（1/0）及到凸包距离。','']
for k in (1,3,5):
    rr=[a for a in r['rows'] if a['K']==k]
    lines += [f'### K{k} 位置','']
    table(['log:GT行','真值坐标','真值内/距','MAP坐标','MAP内/距','MAP来源','均值坐标','均值内/距','GP坐标','GP内/距','同候选GP坐标','同候选GP内/距','最近距离'],[[f"{a['log']}:{a['gt_row']}",xy(a['truth_xy']),f"{int(a['truth_inside'])}/{f(a['truth_hull_distance'])}",xy(a['MAP_xy']),f"{int(a['MAP_inside'])}/{f(a['MAP_hull_distance'])}",','.join(a['MAP_sources']),xy(a['mean_xy']),f"{int(a['mean_inside'])}/{f(a['mean_hull_distance'])}",xy(a['GP_xy']),f"{int(a['GP_inside'])}/{f(a['GP_hull_distance'])}",xy(a['GP_candidates_xy']),f"{int(a['GP_candidates_inside'])}/{f(a['GP_candidates_hull_distance'])}",f(a['nearest_distance'])] for a in rr])
    lines += [f'### K{k} 校准前后分解','']
    table(['log:GT行','阶段','最近/MAP索引','检出最近','检出MAP','未检出最近','未检出MAP','检出Δ','未检出Δ','总Δ','最近秩'],[[f"{a['log']}:{a['gt_row']}",phase,f"{d['correct']}/{d['MAP']}",*[f(d[n]) for n in ('detected_correct','detected_MAP','missing_correct','missing_MAP','detected_delta','missing_delta','total_delta')],d['nearest_rank']] for a in rr for phase,d in a['decomposition'].items()])
    lines += [f'### K{k} 逐点反事实误差','']
    table(['log:GT行','all均值/MAP','S30均值/MAP','S60均值/MAP','oracle均值/MAP'],[[f"{a['log']}:{a['gt_row']}",*[f(a[n+'_restricted_mean_error'])+'/'+f(a[n+'_restricted_MAP_error']) for n in ('all','S30','S60','oracle_hull')]] for a in rr])
lines += ['## 原始文件与执行事实','', '完整结果：[result.json](xacq_f2diag_code_20261111/result.json)。逐点/AP原数：[K1](xacq_f2diag_code_20261111/AP_raw_K1.npz)、[K3](xacq_f2diag_code_20261111/AP_raw_K3.npz)、[K5](xacq_f2diag_code_20261111/AP_raw_K5.npz)。行顺序与 result.json 对应 K 的 rows 相同；列为各文件 ap_ids，包含两类贡献、query_detected、两位置 p、S RSS、全部训练与实际H检出率。result.json 还包含凸包顶点、全部候选位置/来源/凸包关系、逐点时间与索引、输入和源码SHA256。', '', '代码与结果位于探索分支 current_20261110/src/xacq_f2diag_claude_20261111/。已有采用分支、模型、缓存及桥接程序/队列未修改。', '']
lines += ['## 事实小结','', 'K1/3/5 的 22 个真值点中均有 18 个在 F2 训练凸包内，全部 S 在凸包内；三档全部后验均值都在凸包外，MAP 在凸包内仅 4/3/9 点。MAP 坐标来自 F3/F4/F5 的数量分别为 7/2/13、5/2/15、12/2/8，没有 F6 或经验 S。校准后的已检出平均分差仍为 +2.435/+3.027/+2.081，未检出分差为 −8.035/−10.350/−8.775；20/20/15 点的已检出项支持近真值候选，但被未检出项推翻。三档最强未检出负贡献 AP 都是 20:25:00:00:00:83；其 F2/F3 全部训练检出率约 0.331/0.302，F5/F6 为零，22 个 Query 仅 7 点检出，近真值候选校准后平均 p 为 0.653/0.640/0.631。将候选限制在 F2 凸包内后，均值误差由 80.832/78.306/72.478 m 降至 11.430/11.830/10.571 m，仍高于同候选 GP 的 8.067/7.652/7.446 m。S30 为 19.646/43.193/29.099 m，S60 为 40.485/69.024/61.698 m；更多 S 扩大允许区域，简单半径限制并不稳定。事实显示大误差主要随区域外候选获高权重而出现，正确区域内仍有残余误差；本单未证明后者全由排序造成，也未把凸包视为部署可用边界。', '', '运行时间 UTC 04:10:37–04:10:46（北京时间12:10:37–12:10:46）；实际计算8.18秒、峰值RSS209104 KiB、退出码0。runner123518、time父进程123520均已结束。Sol监看者已核实完成。Git经临时Mac定向隧道，代码/轻量结果经SSH同步；没有数据、模型或依赖下载。Git fetch包含协作者轻量结果包；传输计费量未知，不以文件大小冒充网络计量。', '']
target=Path(sys.argv[1]) if len(sys.argv)>1 else P/'report.md'
prefix=os.path.relpath(P,target.parent)
text='\n'.join(lines).replace('xacq_f2diag_code_20261111/',prefix+'/')
target.write_text(text)
