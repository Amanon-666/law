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

本目录包含选定入口递归导入的本地Python源码、轻量证据与[manifest.json](manifest.json)逐文件来源/哈希。源码已做静态语法检查及常见凭据模式检查；此次发布没有运行实验，也没有逐项复验所有服务器运行的源码身份。

数据、权重、缓存数组、第三方包及凭据不在本次上传内。原脚本保留服务器/本机原路径和资产布局；复制src并不能直接运行。旧public_corpus和模型依赖可参阅[此前完整导读](../review_20261109/README.md)及其legacy_public_corpus、legacy_models目录，运行时仍需按真实环境恢复导入路径。未提供虚构的一键复现命令。

[项目总入口](https://app.notion.com/p/3fc217e091ae81d38ed7c81efa4ca70c) · [最新目标与阶段证据](https://app.notion.com/p/406217e091ae81c6b7b9ffb48c08e6a4)

## 01:12 增量同步

九层 K1/3/5 的公开 Tab 与 3NN 比较已补全，见 [完整结果](evidence/trained_system_validation_20261110/complete.json)。公开、普通微调、full24 的 GP+Tab 等权结果分别为 10.9703、10.7880、10.9199 米。

[历史读取归因](evidence/history_training_attribution_20261110/complete.json)补齐普通微调仅 H 的九层结果：公开、普通微调、full24 分别为 13.9782、13.6536、13.5062 米。full24 的 H-only 优势在加入 S 后反转；三者 H+S 为 12.6284、12.1148、12.1917 米。这里的 GP+Tab H-only 仅去掉 Tab 的 S，GP 仍使用 S。H 到 H+S 也改变预处理，不能当作孤立的 S 因果效应。

新增入口：[history_training_attribution_trial.py](src/history_training_attribution_trial.py)。本次文件来源、SHA256 及完成状态见 [增量清单](sync_20261110_0112_manifest.json)。训练增强仍在设计审查阶段，不随本次上传。

## 楼层翻译接入（2026-11-10）

已吸收 Claude 的 `af3c996` 提交，原翻译器和开发痕迹保留在 [vertical_translator_20261110](../vertical_translator_20261110)。本次执行所用的用户压缩包源码原样保留在 [src/vertical_translator](src/vertical_translator)，与协作者后续版本分别保留。

- [vertical_translator_generate.py](src/vertical_translator_generate.py)：B0-only、种子1–5、每种子H/HS模型各一份，247.72秒完成；54组地图及模型留服务器，轻量结果见 [summary](evidence/vertical_translator_b0_ens5_20261110/summary.json)。H/HS翻译图的原3NN均值为11.3927/10.5648米，同表示简单参考图为17.1490米。原型使用B0四层全量源数据，现ordinary30使用三层合法语料。
- [translated_history_context.py](src/translated_history_context.py)：保留连续均值图；按现Tab的−105编码直接作为特征均值，不阈值化或取整，真实S只加入一次。
- [translated_history_system_trial.py](src/translated_history_system_trial.py)：54事件、108次Tab前向已于11:39 UTC完整结束，572.20秒，无Tab训练或GP拟合。预先固定的主输出GP+读学习图的Tab为12.4767米，未改善原GP+ordinary系统10.7880米；同表示简单参考图+Tab组合11.8057米。所有预先声明的读出完整保留在[结果](evidence/translated_history_system_20261110/complete.json)与[冻结](evidence/translated_history_system_20261110/freeze.json)。
- `context_influence_*.py`补齐此前已保存的关系学习实现；v2正式训练仍暂停。旧接线检查对应此前版本，不能当作v2性能结果。

后续代码与关键结果继续同步本研究分支，保留双方提交；原始数据、大模型和凭据不上传。

### 已完成的整体推进

预先声明的另一条读出——学习翻译图的3NN与原GP固定等权——得到 **9.9607米**，较已有GP+ordinary系统10.7880米降低 **7.67%**。K1/3/5分别10.3638/9.8925/9.6258米，均值九层七层改善；同一3NN组合使用未学习参考图为12.8960米。总体位置均值误差p90为20.0522米（原20.7964），但K1为22.9095（原21.1161），少标定尾部仍需解决。

本轮采用原型的四层B0源资产，ordinary30仍是原三层受限源语料；结果是现有资产的系统集成，统一源语料的训练归因留待后续。保留历史翻译及有效GP互补，Tab维持原有原始历史用法；本次均值图替换输入未成功，不据此淘汰Tab或历史翻译机制。

已把有效算法整理为[纯H/S翻译入口](src/frozen_history_translator.py)和[可复用系统入口](src/translated_history_system.py)：加载已有五种子树模型，接收历史、现场标定与待定位扫描，复用原GP输出；不重新训练，不读取整份CSV或Query坐标。两个新入口经过静态审查，以上性能来自冻结的系统评价入口，未再启动数值重跑。原四组正式训练和关系编码器训练继续暂停。

### 输出定义与源监督（未训练）

按用户最新要求，从现象追查学习对象，不把所有组件固定拼进最终方法。平均指纹会合并不同观测机制：50%未检出、检出时−65dBm，与始终−85dBm，按−105编码都得到均值−85；它们对实际扫描的定位证据不同。下一步沿历史翻译主线学习真实单次扫描的检出与RSS分布，统一形成位置证据。旧观测表与条件场已有相关思想，此处保留它们的经验，不另称首次提出概率定位。

[observation_translation_tasks.py](src/observation_translation_tasks.py)已将现43维历史＋12维标定表示接到原共同源任务：H∪S决定AP，只用H/S生成特征，源监督使用另一采集的真实逐扫描RSS并保留未检出值，位置/扫描/AP等权；不再以位置均值充当扫描标签。该模块可复用现有source_episode_catalog/episode_scans，保护源验证位置的权限沿原任务入口执行。仅完成实现与静态审查，尚未拟合概率模型或产生新成绩；当前Tab退步的具体成因仍待这一主线的直接证据。已完成的9.9607米组合保留为实用结果与研究依据，最终结构不要求同时保留GP、Tab和3NN。
