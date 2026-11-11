"""Append execution facts and existing score summaries to the round-four audit."""
import json
from pathlib import Path
base = Path(__file__).parent
s = json.loads((base/'results/bench_round4_summary_20261111.json').read_text())
state = json.loads((base/'results/bench_round4_execution_20261111.json').read_text())
p = base.parent/'bench_round4_20261111.md'
marker = '\n## 执行结果（自动汇总）\n'
text = p.read_text().split(marker)[0] + marker
text += '\n原始位置均值误差合并后计算统计量；static 与 legacy54 分开。表格各分位数均为原协议定义，单位米。当前结果为已暴露开发事件。\n'
for stage in ('smoke', 'dl', 'hfloors', 'legacy54_dl'):
    st = s[stage]
    if not st['completed_buildings']:
        continue
    text += '\n### '+stage+'\n\n已完成建筑 '+', '.join(st['completed_buildings'])+'；预定建筑 '+', '.join(st['expected_buildings'])+'。每个统计单元依次为 mean / p50 / p90。\n\n'
    text += '|范围|方法|完成任务|总体|K1|K3|K5|\n|---|---|---:|---|---|---|---|\n'
    for scope, methods in [('pooled', st['methods']), *st['by_building'].items()]:
        for method, v in methods.items():
            def cell(d):
                return ' / '.join(f'{d[k]:.6f}' for k in ('mean','p50','p90')) if d else '—'
            overall = v['overall']
            text += '|'+scope+'|'+method+'|'+str(overall['completed_tasks'])+'/'+str(overall['tasks'])+'|'+cell(overall)+'|'+ '|'.join(cell(v['by_K'].get(str(k))) for k in (1,3,5))+'|\n'
text += '\n### 时间与发布\n\n|阶段|建筑|耗时秒|Python PID|结果提交|\n|---|---|---:|---:|---|\n'
for r in state['folds']:
    text += '|'+r['stage']+'|'+r['building']+'|'+str(r.get('elapsed_seconds','运行中'))+'|'+str(r['python_pid'])+'|'+r.get('pushed_commit', '同恢复提交a1eeaf0' if r['stage']=='smoke' else '待提交')+'|\n'
text += '\nPyTorch实际版本 `'+s['runtime']['torch_version']+'`。计算仅CPU，线程限制2，实验子进程nice10。源模型joblib只留服务器。源拟合/加载和任务计算耗时分别记录如下。A0源阶段仅加载并序列化已有模型，训练次数为零。DNN任务计算包含H训练及缓存、S适应和Query预测，不能解释为纯推理耗时。\n\n|运行|方法|源拟合/加载秒|任务计算合计秒|选中hp|\n|---|---|---:|---:|---|\n'
for name, run in s['runtime']['runs'].items():
    for method, info in run['source_fit'].items():
        hp = (run['selected_hp'] or {}).get(method,{}).get('hp')
        text += f"|{name}|{method}|{info.get('seconds')}|{run['prediction_seconds'].get(method)}|{hp}|\n"
text += '\n完整hp_scores、模型SHA256、逐任务耗时、代码哈希与调用命令保存在相应JSON。\n\n### 修正与时间顺序\n\n- 方法修正为审查时加入缓存坐标下界与尺度键，防止相同AP轴而Support范围变化时复用错误坐标系统。正式运行前完成。\n- 原始smoke顺序为pretrain、finetune、MAML，与请求前两项顺序不同。原smoke耗时210.949479秒，三个事件均成功。\n- smoke完成后的元数据提取因driver缺少law_bench导入路径失败。随后只修正runner的sys.path，以--resume跳过已完成任务，元数据恢复耗时0.466981秒，没有重训。首条失败记录及traceback保留。resume将原smoke的3行stdout覆盖为空；原始source/event JSON完整。\n- 55分钟截止时间沿用首次smoke开始2026-11-11 00:25:15.853 UTC，因此01:20:15.853 UTC停止启动后续任务，正在运行的进程保持。\n'
p.write_text(text)
