# UJI 少标定定位 · 重构分支

本分支面向楼层已知、同楼历史指纹已配准的少标定定位。

## 当前入口

本分支以“少量现场标定怎样修正历史指纹”组织实现。默认入口为 `refi.fit_map(histories, support)`，得到可重复调用 `.predict(query_rss)` 的目标楼层指纹图。默认模型只依赖 NumPy、SciPy 和已有的 TPM 指纹图适配代码。训练和评测脚本均在 `research/uji_core/`，部署无需读取该目录。

- [方法取舍、三项目标的实测结果和复现命令](research/uji_core/REPORT.md)
- [部署代码](refi.py) · [UJI 抽样定义](research/uji_core/protocol.py) · [配对结果](research/uji_core/results/building_means.csv)

```python
from refi import fit_map

# histories 中每一项和 support 均为 {"rss": [N, A], "xy": [N, 2]}。
# 每次试验内各矩阵 AP 顺序一致，100 表示未检出，坐标在同一水平参考系。
target_map = fit_map(histories, support)
xy = target_map.predict(query_rss)  # 多次查询复用同一张图
```

维护这条主线时，初稿作为研究方向。新设计进入默认路径前，需要在同样的现场数据、同样的读出和强 WKNN 之下，单独检验它带来的定位增量。下方为分支建立前的研究归档。

## 历史代码范围

- 根目录：概率观测模型、历史指纹先验、Support 适应、全扫描定位，以及历史实验入口。
- `residual_probe/`：历史条件残差学习、同特征现场拟合、固定查询与嵌套 K 对照；本轮入口为 `broad_source.py` 和 `broad_evaluation.py`。
- `public_corpus/`：公共指纹语料读取与任务构造。
- `external_baselines/`：成熟方法的本项目适配实现；不是原作者官方代码。
- `operator_prior_v1/`、`learnable_profile/`、`physical_prior_candidate/`：保留的历史候选及诊断实现，不代表均属于当前主模型。

## 原 main 的研究记录

截至 2026-10-27，仅扩大源训练任务覆盖的候选已完成开发评价并停止：在 K=1/3/5 的楼内平均定位误差比较中，只有 2/4 个建筑同时优于成熟历史基座和同特征现场拟合，未通过预先规定的学习增量门槛。胜过 H+S WKNN 的结果不能单独证明学习更有效地利用历史。共 57 个完整评价事件，另有 3 个 Pisa K=20 事件未通过数值收敛门槛。独立验证未启动；本仓库不宣称已证明泛化收益或方法创新。

ScaML-GP 比较已接入：`scaml_run.py` 保留官方重启选择行为；`scaml_training_selection.py` 是单独的事后修正对照，恢复训练模式后再选择重启，避免将条件后验预测评分误作训练目标。两者共用冻结的 H/S/Q/AP、编码、核、先验和预算，分别保存结果。两版开发比较均已完成 60/60 个设置并核验冻结身份与逐查询误差。修正会改变部分成绩，但在当前四建筑的各 K 三划分均值上，均未超过 H+S WKNN 或已取得结果的历史基座；这只评价本次高维 RSS→坐标适配，不证明原机制无效或本项目具有创新。

已停止学习候选的固定源样本诊断也已完成：最终参数相对零参数的四折总训练目标估计均小幅下降，但各折仍有来源/K 分组变差。该有限积分诊断不是定位收益、全局收敛或泛化证明，不改变候选停止结论。

## 运行条件

这是现有工作副本的代码归档，不是已经验证可一键复现的发行包。主要依赖为 Python、NumPy、SciPy、PyTorch 和 pypmc；绘图使用 Matplotlib。不同历史入口的依赖和数据路径以代码为准。

数据、训练权重、历史缓存、冻结实验资产和环境不包含在仓库内。部分入口依赖原实验目录 `results/` 中的准备结果或代码快照，以及相邻项目中的数据/依赖；部分 shell 脚本保留原服务器路径。运行前需提供具有使用权限的本地数据与对应资产、核对路径和依赖。请勿把缺失资产替换为不同数据后仍声称复现原冻结实验。

上传保留现有算法代码，未因发布而改动模型、参数或训练流程。未附加新的开源许可证。

ScaML 入口依赖 Bosch 官方代码提交 `6717858eeacbe5d310bfe40447993f75aa2ad748`（[仓库](https://github.com/boschresearch/Scalable-Meta-Learning-with-Gaussian-Processes)，AGPL-3.0），本仓库不包含该源码。实际运行使用 BoTorch 0.7.3、GPyTorch 1.9.0、linear-operator 0.2.0、blackboxopt 5.0.3、parameterspace 0.10.0，复用 Python 3.12 / PyTorch 2.8；这不是作者原锁文件环境。协议适配与选择修正均应明确标注，不能称作者原始无线定位实验复现。
