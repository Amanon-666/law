# LAW: history-conditioned Wi-Fi localization

研究代码快照。目标是利用多环境历史指纹语料 H 和新环境少量带位置标签的扫描 S，提高新环境的定位精度。部署时仍保留历史库，语义入口为 `predict(history_bank, support, query_scan)`。

## 代码范围

- 根目录：概率观测模型、历史指纹先验、Support 适应、全扫描定位，以及历史实验入口。
- `residual_probe/`：历史条件残差学习、同特征现场拟合、固定查询与嵌套 K 对照；本轮入口为 `broad_source.py` 和 `broad_evaluation.py`。
- `public_corpus/`：公共指纹语料读取与任务构造。
- `external_baselines/`：成熟方法的本项目适配实现；不是原作者官方代码。
- `operator_prior_v1/`、`learnable_profile/`、`physical_prior_candidate/`：保留的历史候选及诊断实现，不代表均属于当前主模型。

## 当前证据边界

截至 2026-10-27，仅扩大源训练任务覆盖的候选已完成开发评价并停止：在 K=1/3/5 的楼内平均定位误差比较中，只有 2/4 个建筑同时优于成熟历史基座和同特征现场拟合，未通过预先规定的学习增量门槛。胜过 H+S WKNN 的结果不能单独证明学习更有效地利用历史。共 57 个完整评价事件，另有 3 个 Pisa K=20 事件未通过数值收敛门槛。独立验证未启动；本仓库不宣称已证明泛化收益或方法创新。

ScaML-GP 比较已接入：`scaml_run.py` 保留官方重启选择行为；`scaml_training_selection.py` 是单独的事后修正对照，恢复训练模式后再选择重启，避免将条件后验预测评分误作训练目标。两者共用冻结的 H/S/Q/AP、编码、核、先验和预算，分别保存结果。目前完整比较仍在运行，不据部分结果宣称方法优劣。

## 运行条件

这是现有工作副本的代码归档，不是已经验证可一键复现的发行包。主要依赖为 Python、NumPy、SciPy、PyTorch 和 pypmc；绘图使用 Matplotlib。不同历史入口的依赖和数据路径以代码为准。

数据、训练权重、历史缓存、冻结实验资产和环境不包含在仓库内。部分入口依赖原实验目录 `results/` 中的准备结果或代码快照，以及相邻项目中的数据/依赖；部分 shell 脚本保留原服务器路径。运行前需提供具有使用权限的本地数据与对应资产、核对路径和依赖。请勿把缺失资产替换为不同数据后仍声称复现原冻结实验。

上传保留现有算法代码，未因发布而改动模型、参数或训练流程。未附加新的开源许可证。

ScaML 入口依赖 Bosch 官方代码提交 `6717858eeacbe5d310bfe40447993f75aa2ad748`（[仓库](https://github.com/boschresearch/Scalable-Meta-Learning-with-Gaussian-Processes)，AGPL-3.0），本仓库不包含该源码。实际运行使用 BoTorch 0.7.3、GPyTorch 1.9.0、linear-operator 0.2.0、blackboxopt 5.0.3、parameterspace 0.10.0，复用 Python 3.12 / PyTorch 2.8；这不是作者原锁文件环境。协议适配与选择修正均应明确标注，不能称作者原始无线定位实验复现。
