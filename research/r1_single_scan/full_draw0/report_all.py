"""Render a narrow B1-only report; never infer success on manuscript or other buildings."""
import csv,json,sys
from pathlib import Path
from statistics import mean
A=('gp','seven','nonlinear');N={'gp':'原历史 GP','seven':'冻结七系数修正','nonlinear':'冻结非线性 V1','fixed_half':'既有固定半平滑轻量法','empirical':'同源选参的实测 H+S WKNN','gp_map':'原 GP 图 → WKNN','seven_map':'七系数图 → WKNN','nonlinear_map':'非线性图 → WKNN','measured_prototype_fixed3':'实测位置原型固定 3-WKNN'}; KS=(1,3,5,10,20);RR={'official':'官方后期查询','phone':'严格留手机查询'}
def main(inp,out):
    s=json.loads(inp.read_text());out.mkdir(exist_ok=True,parents=True);text=[];line=text.append
    line('# B1 每处单扫：原十次标定抽样、五档预算、同图近邻')
    line(f"实际复核 **{s['verified']}/400 个 B1 事件**。每条记录是一个楼层、查询协议、标定顺序及预算，而不是一栋独立建筑。原 GP、七系数和非线性全程冻结，不随预算或查询切换。未使用新外部测试或额外40抽；本实验不是原稿表 II 的 K5R3，数值为 UJI 原坐标 U，不是米。")
    line('## 同协议定位误差（U，越低越好）')
    for r,lab in RR.items():
        line('### '+lab);line('| 方法 | K=1 | K=3 | K=5 | K=10 | K=20 |');line('|---|---:|---:|---:|---:|---:|');cv=s['curves'][r]
        for n in (*A,'fixed_half','empirical',*((('measured_prototype_fixed3',) if r=='official' else ())),*(a+'_map' for a in A)):
            line('| '+N[n]+' | '+' | '.join(f"{cv[str(k)][n]:.3f}" if str(k) in cv else '缺失' for k in KS)+' |')
        line('');line('各楼层十抽等权后，再对四层等权。同图比较使用同一学习臂的观测化后验图（包括 AP、候选坐标），源端锁定的邻居数各为 1/7/3。实测 H+S WKNN 使用历史与标定的实测库，不是同图；固定半平滑是已有同部署信息的端到端固定方法，网格、建图和读出不完全相同，不作单组件归因。')
        if len(cv)==5:
            for n in A:
                ds=[(u,v,cv[str(v)][n]-cv[str(u)][n]) for u,v in zip(KS[:-1],KS[1:])];fails=[f'{u}→{v} ({dv:+.3f} U)' for u,v,dv in ds if dv>=0];floorwin=[f for f in range(4) if all(s['floor_means'][r][str(v)][n][f]<s['floor_means'][r][str(u)][n][f] for u,v in zip(KS[:-1],KS[1:]))];v=sum(cv[str(k)][n]<cv[str(k)][n+'_map'] for k in KS);emp=sum(cv[str(k)][n]<cv[str(k)]['empirical'] for k in KS);fl=sum(s['floor_means'][r][str(k)][n][f]<s['floor_means'][r][str(k)][n+'_map'][f] for k in KS for f in range(4));line(f"- {N[n]}：四层平均的相邻预算 {4-len(fails)}/4 下降；未下降：{', '.join(fails) or '无'}。逐层每一步都降的楼层：{floorwin}。低于自己的同图 WKNN {v}/5 个建筑×预算、{fl}/20 个楼层×预算；低于实测库源端选参 WKNN {emp}/5 个建筑×预算。")
    line('## 学习的实际增量，与读出方式区分')
    line('新方法减对照，负数为好。以下区间在每个楼层按共同 8 U 空间块与原十个标定顺序配对重抽 4,000 次；不同 K 共享抽样。仅是既定楼、既定两套查询的条件区间，不是多重比较校正后的跨楼证据。')
    line('| 查询 / K | 新 − 对照 | 差 U | 变化 % | 条件 95% 区间 | 改善楼层 / 4 | 改善抽样 / 10 |');line('|---|---|---:|---:|---:|---:|---:|');contr=[('seven-gp','七系数 − GP'),('nonlinear-gp','非线性 − GP'),('nonlinear-seven','非线性 − 七系数'),('gp-gp_map','原 GP 概率读出 − 同图 WKNN'),('seven-seven_map','七系数概率读出 − 同图 WKNN'),('nonlinear-nonlinear_map','非线性概率读出 − 同图 WKNN')]
    for r,lab in RR.items():
        for k in (5,20):
            if str(k) not in s['paired'][r]:continue
            for key,label in contr:
                p=s['paired'][r][str(k)][key];l,h=p['ci95'];line(f"| {lab} / {k} | {label} | {p['delta']:+.3f} | {p['pct']:+.2f}% | [{l:+.3f}, {h:+.3f}] | {p['floor_wins']}/4 | {p['draw_wins']}/10 |")
    line('## 大误差与失败楼层')
    for r,lab in RR.items():
        for k in (5,20):
            if str(k) not in s['paired'][r]:continue
            pp=s['p90'][r][str(k)];line(f"- {lab} K{k}：GP / 七系数 / 非线性 P90 = {pp['gp']:.3f} / {pp['seven']:.3f} / {pp['nonlinear']:.3f} U。先取每事件位置权重下的扫描误差第90百分位，再平均，不是把所有事件的扫描拼起来取分位。")
            for n in ('seven','nonlinear'):
                p=s['paired'][r][str(k)][n+'-gp'];line(f"  - {N[n]}减 GP，F0/F1/F2/F3："+' / '.join(f'{v:+.3f}' for v in p['floor_delta'])+' U。')
    o=s['optimizer'];hs=[v for k,v in o['history'].items()];line('## 数值、成本和边界');line(f"- B1 现场各热点后验总计 {o['site_AP_arm_calls']} 次；SciPy 退出标志非成功 {o['unsuccessful_exit']} 次；所有现场求解记录中的最大投影一阶残差 {o['max_projected_grad']:.3g}。每条警告均保留，小梯度不等于全局最优证明。源端邻居选参仅用 B0F1/B2F2 两个开发场景，不称穷尽所有 WKNN。")
    line(f"- 各分片建图与评分耗时加总 {s['cpu_shard_elapsed_sum_seconds']/3600:.2f} 单线程工作小时，不含安装及复核，非单用户查询时延。K5、K10、K20 的已归档 F0 官方 draw0 逐坐标桥接见两抽复核，不能把旧 M-PMC 与 VI 差当学习增益。")
    line(f"- 缺失格 {len(s['missing'])}。仅在40个楼层×标定顺序齐全时列出该协议该预算的十抽建筑均值。每组原始坐标、行号、候选网格摘要、优化终点、各楼与实际失败原因均可复算。")
    if s['missing']:line('失败格：'+json.dumps(s['missing'],ensure_ascii=False))
    line('不能将官方与留手机两列不同的赢家拼成系统；这两套查询和历史/标定样本的构造不同，不能把排名变化纯归因于设备。只证明 B1 单扫，不代表三建筑、原稿 K5R3 或独立外测。未进入 B1 训练、但反复查过的 B1 仍是开发楼。外部预留不打开。')
    line('[预先登记的全十抽协议](../../TEN_DRAW_PROTOCOL.md) · [计算与归档](https://github.com/Amanon-666/law/actions/runs/41563449591) · [同一目录机器结果](ten.json)')
    (out/'REPORT_B1_R1_ORIGINAL_TEN.md').write_text('\n\n'.join(text).replace(' |\n\n|',' |\n|').replace('|\n\n|','|\n|')+'\n')
    with (out/'curve.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['regime','K','method','MDE_U','macro_P90_U']);
        for r,v in s['curves'].items():
            for k,m in v.items():
                for n,d in m.items():w.writerow([r,k,n,d,s['p90'][r][k].get(n,'')]);
    with (out/'paired.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['regime','K','new_minus_base','delta_U','relative_pct','conditional_lo','conditional_hi','floor_wins','draw_wins','F0','F1','F2','F3','delta_original_draws_2_to_9']);
        for r,v in s['paired'].items():
            for k,m in v.items():
                for n,d in m.items():w.writerow([r,k,n,d['delta'],d['pct'],*d['ci95'],d['floor_wins'],d['draw_wins'],*d['floor_delta'],d['new8_delta']]);
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
