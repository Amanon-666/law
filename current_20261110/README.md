# LAW 当前主线源码（2026-11-10）

这是当前本机工作副本的发布快照，保留源文件内容，不为展示重写算法。GP历史场推断、训练后的Tab及已有有效组合均是方法内部组成部分；当前研究重点是改进训练框架的历史利用与目标迁移能力。

## 阅读顺序

| 功能 | 入口 |
| --- | --- |
| 已有完整源任务训练（约6.77米源端结果） | [tabicl_multitask_full_trial.py](src/tabicl_multitask_full_trial.py) |
| 普通微调及目标归因 | [ordinary_context_trial.py](src/ordinary_context_trial.py)、[ordinary_context_target_trial.py](src/ordinary_context_target_trial.py) |
| 原始历史读取与学习器 | [tabicl_history.py](src/tabicl_history.py)、[tabicl_history_learning.py](src/tabicl_history_learning.py) |
| 九层共同H/S/Q与历史资产 | [building_information_protocol.py](src/building_information_protocol.py)、[building_information_assets.py](src/building_information_assets.py) |
| GP完整定位路径 | [building_information_gp_trial.py](src/building_information_gp_trial.py)、[position_fusion_gp.py](src/position_fusion_gp.py) |
| 两路分布融合及简单组合比较 | [position_fusion_trial.py](src/position_fusion_trial.py)、[position_distribution_fusion.py](src/position_distribution_fusion.py) |
| 稳定性分析 | [system_stability.py](src/system_stability.py) |
| UAH外部评价 | [uah_system_assets.py](src/uah_system_assets.py)、[uah_system_trial.py](src/uah_system_trial.py) |
| K3/K5公开模型及3NN补全 | [trained_system_validation_trial.py](src/trained_system_validation_trial.py)；已完成，见对应complete.json |

## 已有证据与当前判断

[evidence](evidence)保存本机已有轻量结果，具体协议及分组读数以各complete.json为准：
- 源建筑开发：公开Tab约9.70→完整任务训练6.77米；普通微调约8.27米。
- 目标任务：普通微调约12.11米，原任务训练约12.19米；特定任务组织的源端优势尚未迁移。
- UJI九层K1/3/5：GP约11.90米、分布融合10.82米、原强Tab简单等权10.92米；普通微调Tab简单等权约10.79米。
- UAH单楼层双路线：GP约4.64米，组合4.61米，尾部亦有改善。

不同协议数字不构成一条训练进度曲线。简单组合已有效，复杂融合不自动有独立贡献。失败探索没有作为当前主线入口；必要历史实现和此前审阅材料保留在仓库的review_20261109目录。

## 快照与运行边界

本目录包含选定入口递归导入的42份本地Python源码、轻量证据与[manifest.json](manifest.json)逐文件来源/哈希。源码已做静态语法检查及常见凭据模式检查；此次发布没有运行实验，也没有逐项复验所有服务器运行的源码身份。

数据、权重、缓存数组、第三方包及凭据不在本次上传内。原脚本保留服务器/本机原路径和资产布局；复制src并不能直接运行。旧public_corpus和模型依赖可参阅[此前完整导读](../review_20261109/README.md)及其legacy_public_corpus、legacy_models目录，运行时仍需按真实环境恢复导入路径。未提供虚构的一键复现命令。

[项目总入口](https://app.notion.com/p/3fc217e091ae81d38ed7c81efa4ca70c) · [最新目标与阶段证据](https://app.notion.com/p/406217e091ae81c6b7b9ffb48c08e6a4)

## 01:12 增量同步

九层 K1/3/5 的公开 Tab 与 3NN 比较已补全，见 [完整结果](evidence/trained_system_validation_20261110/complete.json)。公开、普通微调、full24 的 GP+Tab 等权结果分别为 10.9703、10.7880、10.9199 米。

[历史读取归因](evidence/history_training_attribution_20261110/complete.json)补齐普通微调仅 H 的九层结果：公开、普通微调、full24 分别为 13.9782、13.6536、13.5062 米。full24 的 H-only 优势在加入 S 后反转；三者 H+S 为 12.6284、12.1148、12.1917 米。这里的 GP+Tab H-only 仅去掉 Tab 的 S，GP 仍使用 S。H 到 H+S 也改变预处理，不能当作孤立的 S 因果效应。

新增入口：[history_training_attribution_trial.py](src/history_training_attribution_trial.py)。本次文件来源、SHA256 及完成状态见 [增量清单](sync_20261110_0112_manifest.json)。训练增强仍在设计审查阶段，不随本次上传。
