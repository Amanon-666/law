# LAW / ReFi 实际实现审阅快照

2026-11-09。给第一次接手项目的研究者：这里提供实际源码，不是把理想方案重新写成代码。没有运行新实验，也没有为了发布重构算法。`src/` 是本机恢复工作副本中选定入口的递归本地 Python 依赖；`evidence/` 是已有轻量结果和部分冻结记录。`manifest.json` 保存逐文件来源、SHA256、本地导入图和可核验的实验源码对应。

**当前任务：借鉴 GP 少标定优势，使历史组织、跨环境学习与目标适应框架更有效，以完整训练后的 Tab 为主要实例。**可靠少标定、多源幅度/空间/覆盖处理、独立学习增量三个目标同时保留。下面各条代码路径不是已经整合成功的一套方法。

[Notion 总入口](https://app.notion.com/p/3fc217e091ae81d38ed7c81efa4ca70c) · [当前总览](https://app.notion.com/p/407217e091ae8110b7ecc4868282f643) · [阶段记录](https://app.notion.com/p/406217e091ae81c6b7b9ffb48c08e6a4)

## 先读哪些代码

| 要检查的问题 | 入口与下一层 | 状态 |
|---|---|---|
| Tab 的历史与标定究竟怎么输入 | [tabicl_history.py](src/tabicl_history.py) → `features`、`FrozenHistoryRegressor.fit/predict` | 当前强工具/对照 |
| 9.70→6.77 是怎样训练出来的 | [tabicl_multitask_full_trial.py](src/tabicl_multitask_full_trial.py) → [tabicl_history_learning.py](src/tabicl_history_learning.py) → [source_episode_catalog.py](src/source_episode_catalog.py) | 已完成源训练 |
| 源收益怎样迁到 B1 | [tabicl_multitask_transfer.py](src/tabicl_multitask_transfer.py) → [tabicl_context_matching_trial.py](src/tabicl_context_matching_trial.py) | 已完成开发评价，未胜最强 GP |
| 原稿 ReFi 到底实现了什么 | [recomposition.py](src/recomposition.py) → [posterior.py](src/posterior.py)、[observations.py](src/observations.py)；实验入口 [recomposition_trial.py](src/recomposition_trial.py) | 已实现的原稿实例，非当前 Tab 本体 |
| 强 GP 如何校准和保留跨采集信息 | [fields.py](src/fields.py)、[adaptation.py](src/adaptation.py)、[scalar_gp_adaptation.py](src/scalar_gp_adaptation.py)、[acquisition_transfer.py](src/acquisition_transfer.py)、[acquisition_retention_trial.py](src/acquisition_retention_trial.py) | 不同协议使用不同适应/读出，勿混用 |
| 11.96→10.79 的历史重组改了什么 | [coupled_scan_transport.py](src/coupled_scan_transport.py) → [coupled_scan_transport_trial.py](src/coupled_scan_transport_trial.py) | 固定 Tab、K1 局部收益 |
| 修正分配、生成指纹为何未改善 | [common_update_transport.py](src/common_update_transport.py)、[prior_mean_transport.py](src/prior_mean_transport.py)、[predictive_scan_context.py](src/predictive_scan_context.py)，各自 trial | 完成的探索，未采用 |
| 固定 Tab 表示再接 GP 是否已做过 | [history_feature_gp_trial.py](src/history_feature_gp_trial.py)、[history_feature_gp.py](src/history_feature_gp.py)、[history_feature_cache.py](src/history_feature_cache.py) | 条件学习几乎无独立收益 |
| 最新 GP 排序的正结果从哪里来 | [history_position_ranking.py](src/history_position_ranking.py)、[history_position_ranking_trial.py](src/history_position_ranking_trial.py) | 平均改善但跨楼不稳，非 Tab 学习成果 |
| 跨源共享排序实现到了哪 | [source_position_ranker.py](src/source_position_ranker.py)、[source_position_ranker_trial.py](src/source_position_ranker_trial.py)、[source_gp_boundary.py](src/source_gp_boundary.py) | 源准备失败，未得到新共享模型；后续修复未获性能验证，不作为当前主线 |

## 数据权限与实际表示

H 是带位置、来源身份的历史扫描，S 是目标少量带位置扫描，Q 是待定位扫描。K 指位置数，R 指每位置重复扫描。当前同址历史需要坐标/AP对应，且目标楼层已知。其他建筑的学习经验不等于其地图可以直接按坐标/AP拼进目标。

`uji.py`、`acquisition_tasks.py`、`source_episode_catalog.py`、`paired_acquisition_tasks.py` 是数据与分组入口。源目录按 `(USERID, PHONEID, UTC日期)` 定义采集组；这会混杂用户、设备和日期，不能解释成独立设备因果效应。源验证预留位置在全局允许历史和支持中排除；实际任务间仍共享楼层、采集组和数据，27事件不是27独立环境。请核对代码的集合条件，而非只相信注释。

Tab 的 AP 集是 H 与当前 S 中曾检出 AP 的并集，矩形边界也来自 H/S。Query 不用于拟合这些统计。原始 `100` 表示未检出；Tab 将其编码成0，已检出值映射为 `(RSS+105)/105`，再加物理楼层一列。它没有独立的显式检出掩码特征；极限信号值和未检出编码可能需要审视。原始行数、采样密度、来源楼层分布均进入上下文，并未默认均衡来源。

`FrozenHistoryRegressor.fit` 拼接所有 H 行和真实 S 行，以所有上下文坐标均值为原点；两个轴各建一个回归适配器，但共享同一个底层模型。每轴8个预处理视图，官方预处理/目标标准化继续存在。`predict` 输出原始坐标与按H/S矩形裁剪的坐标，正式比较通常用后者。不要把这个直接坐标路径解释成GP位置后验。

## Tab 源训练和部署的完整链

1. `events_for` 固定24个联合更新，每步3个源楼层任务，共72事件；K1/3/5分布固定。源验证27事件，固定draw2，fit使用不同draw；来源/查询保护由 `pack` 和目录检查约束。
2. 每个任务调用相同 `fit(H,S)`，只拟合上下文处理；共享模型与优化器连续。`views` 用已拟合的处理转换查询RSS，查询坐标不进入特征拟合。
3. `EpisodicHistoryRegressor.backward` 对两个坐标轴、8视图、999分位数使用官方pinball损失；相同位置多扫描按位置等权。`embed_with_test=False`。这不是ReFi的网格位置NLL，也不是直接最小化欧氏距离。
4. 完整可学习参数约2854万，固定的8个旋转频率不解冻。AdamW学习率1e-5、weight decay .01、梯度裁剪1；GPU混合精度与recompute。每步三任务梯度平均后更新。
5. 记录12/24步，只有24步是预定正式结果，不按目标结果挑checkpoint。共享模型重新加载后评分，源验证未更新权重。
6. 目标部署固定full24，只重新构造目标H/S上下文，没有目标查询梯度更新。B1结果是已经多次查看的开发任务，不能称独立盲测。

值得质疑的地方：源训练任务是否逼近实际目标变化？源内位置隔离是否足以代表跨楼迁移？两坐标轴共享模型但分别预测是否损失联合几何？楼层特征能否在重组后仍保持语义？pinball与最终MDE/尾部是否匹配？这些是审查问题，不是本次已确认的错误。

## 原 ReFi 12 参数和 GP 路径

`Parameters.values` 给出9维门控、η=.5*tanh(θ)、有界log均值/方差尺度。局部门控以原全局w乘指数得分归一；原w=0的来源永远不会被局部门控复活。特征包括几何覆盖、方差、缺历史、来源信号差/形状差及楼层差交互。`Source.block` 用η的二次式计算同一T作用下的协方差。

`prior` 在标定位置、候选到标定交叉块及候选边际上一致构造均值/协方差；均值围绕固定中心−75调幅，历史核乘s²，残差和截距另加。全部缺历史的AP绕过结构参数。覆盖外外推、部分缺历史回退和全缺历史回退是不同分支，不能据一般公式忽略。

`ImportanceSupport` 用实际保存的提议样本与原始提议密度重新加权，标定后验随当前参数变化。ESS门槛是数值条件，不是性能保证。`predict_from_posterior` 将S后验矩传播到候选位置；`scan_log_prob` 同时处理RSS和未检出，按AP求和并softmax得到位置概率。`evaluate` 用最近网格位置NLL训练，输出坐标期望评价欧氏误差；两遍反传是降低内存的实现，不是另一训练目标。

强GP不是一个无歧义的文件名：旧多点适应用重要性提议/分阶段参数估计，近期K1用 `ScalarSupportIntegral` 和权重/截距profile；读出可能是均值或几何中位数。`acquisition_transfer` 将历史信息与采集扰动传播区分：改变跨采集的协方差块，S边缘和目标新采集方差保持，retention=0并不等于完全不用S。不同实验必须看对应trial所选控制，不能把更弱读出称为唯一强GP。

## 历史重组到底改了哪些数

`coupled_scan_transport.build_transport` 只支持K1、1或2来源，使用已选的GP权重和截距；各AP内来源与位置联合条件化，采样8个独立历史“世界”，逐来源把已检出真实RSS加上 `融合条件场−该来源条件场`。保留原扫描内AP配对和原未检出掩码；不同AP的GP场仍独立。真实S由消费者每次插入一次。单来源差值恒零；不会重建旧历史没有检出的目标信号。

`common_update_transport` 加入GP共同更新；`prior_mean_transport` 则使用 `原RSS−H-only来源均值＋目标融合均值＋共同更新`，假定历史经验残差可迁移。这个假设并非由一个标定点识别。`_finish` 记录超出参考RSS范围的值，不裁剪RSS。多世界需要多次Tab推断，不能只报均值收益不报成本。

`predictive_scan_context` 是另一条GP生成观测再交给Tab的路径，不能混入“真实扫描重组”结果。原固定配置四任务退步；完整观测信息进入输入不等于消费者保住原GP能力。

## GP 排序分支暴露了什么

`history_position_ranking` 使用成熟XGBoost pairwise排序：每条历史扫描组成1个历史位置正例和4个负候选；以GP评分作为base_margin，并把GP评分提供给所有对照。每次现场拟合100棵树、深度6。预测返回树残差后加回GP一次。它是位置判别，不是可靠的历史密度比或校准概率；充分拟合时可以抵消GP起点。

原始与校准历史、GP生成样本、取消offset等对照见trial。最新11.52对12.07是16次现场排序拟合所得，不是跨源共享网络训练，也不是Tab被增强。source_position_ranker想进一步使用真实源任务监督，但准备失败后未获得正式训练结果。不要用新源码存在代替成功。

## 分数到原记录的对应

| 结果 | 原始轻量记录 |
|---|---|
| 源公开9.698→full24 6.7719，WKNN11.4547 | [source complete](evidence/tabicl_multitask_full_trial_20261107_checked/complete.json)、同目录freeze/training/step_24 |
| B1 K5公开10.4068→10.0250；retained GP9.9406 | [transfer complete](evidence/tabicl_multitask_transfer_20261107/complete.json) |
| K1固定Tab11.9592→10.7866，简单均值10.8954 | [transport complete](evidence/coupled_scan_transport_20261108_numeric_replay/complete.json) |
| 固定表示GP条件7.3068，普通7.3080，完整Tab6.7719 | [feature GP complete](evidence/history_feature_gp_trial_20261107/complete.json) |
| 修正分配替换未解决跨楼层失效 | [prior transport complete](evidence/prior_mean_transport_20261108/complete.json) |
| GP生成补充四任务均退步 | [predictive context complete](evidence/predictive_context_20261108/complete.json) |
| GP排序11.52对12.07，一楼改善一楼退步 | [ranking complete](evidence/history_position_ranking_fixed_20261109/complete.json) |
| 跨源排序准备失败，未训练 | [source ranking failure](evidence/source_position_ranker_20261109/complete.json) |

不同K、查询集合、读出和任务不能拼成一条曲线。源位置保留验证不等于新建筑独立验证。完整原始逐扫描预测/权重未发布，轻量JSON足以审查记录与归因，不足以独立重算全部分数。

## 快照身份、依赖和缺口

`manifest.json`中已有63项可对齐的源码SHA256与所附freeze记录一致；未附freeze的结果只保留其原hash引用，不冒称本次核验了全部代码版本。该清单证明所复制内容的身份，不证明当前服务器工作区逐文件一致。当前主线程仍可继续修改代码；请引用提交SHA而非浮动分支。

`legacy_models/`两文件与paired_assets_20261106冻结哈希相同，来自本机相同字节副本。`legacy_public_corpus/`保留旧public_corpus入口的项目代码依赖，未重新证明与每次旧服务器运行一致。原加载路径未改，因此它们不是开箱即用的安装布局。

第三方TabICL、XGBoost、PyTorch等不在此重新发布；以源码import、freeze记录的包文件hash和原作者发布版本为准。TabICL使用项目2.2.0、regressor-v2-20260212；未把它当作自研网络。[作者源码](https://github.com/soda-inria/tabicl)。XGBoost是成熟排序工具，不是本项目创新。

保留的原代码含服务器绝对路径、相邻项目相对路径、冻结输入hash、主机限制以及中间缓存依赖。数据CSV、历史场缓存、源任务目录、原checkpoint与逐查询数组不随包发布。**这是可逐行审查的源码快照，不是已验证一键复现包。**不要直接运行trial；它们可能创建实验目录并训练，缺资产时也不能替换输入后仍称原实验复现。

源码里check模块被实验入口导入、多个trial互相复用辅助函数、目录路径耦合等都原样暴露。读者可以据此批评实现结构；不能仅因检查多、梯度非零或代码能运行就认定科研逻辑闭合。

## 希望外部审查得出什么

请从输出逆推训练、适应、查询各阶段实际读到的数据和更新的状态，明确指出不自然的接口、信息丢失、不可识别假设或被对照混淆的贡献。区分已确认硬伤、合理疑点和有限失败。允许推翻当前实现设计，不必为GP、Tab或原稿12参数辩护；但应保留真实积极结果，并提出一个围绕历史利用与少样本适应的完整破局判断。
