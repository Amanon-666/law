# LAW 当前主线源码（2026-11-10）

当前采用 **Claude原版五种子观测分布翻译＋既有GP位置证据互补**：原54个开发任务中平均 **8.7792米**、总体p90 **15.9539米**。原版训练、源语料、抽样、扫描计数权重和五模型集成完整保留；两路在原候选位置上固定等权结合对数证据，再输出位置。相比同一原版单路9.4325米改善6.93%，也低于此前种子11–15的9.3664米。GP、训练后的Tab、9.9607米旧组合及所有正负结果保留为可用资产，不要求同时进入最终系统。

本项目10.0337米实现同时改变了多项训练条件，身份是受限比较，尚未超过最好版本。旧划分用于特定归因，不自动约束所有正式方法。原版发布脚本未保存权重；[原配方资产生成](src/claude_hurdle_assets.py)和[H/S-only部署入口](src/frozen_hurdle_translator.py)用于把最好方案实际接续，原样源码在[src/claude_hurdle_20261110](src/claude_hurdle_20261110)。

## 阅读顺序

| 功能 | 入口 |
| --- | --- |
| 当前采用：原版五模型训练资产与直接部署 | [claude_hurdle_assets.py](src/claude_hurdle_assets.py)、[frozen_hurdle_translator.py](src/frozen_hurdle_translator.py) |
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

已吸收 Claude 的 `af3c996` 及 `e1a9983` 提交，原翻译器和开发痕迹保留在 [vertical_translator_20261110](../vertical_translator_20261110)。本次执行所用的用户压缩包源码原样保留在 [src/vertical_translator](src/vertical_translator)，与协作者后续版本分别保留。

- [vertical_translator_generate.py](src/vertical_translator_generate.py)：B0-only、种子1–5、每种子H/HS模型各一份，247.72秒完成；54组地图及模型留服务器，轻量结果见 [summary](evidence/vertical_translator_b0_ens5_20261110/summary.json)。H/HS翻译图的原3NN均值为11.3927/10.5648米，同表示简单参考图为17.1490米。原型使用B0四层全量源数据，现ordinary30使用三层合法语料。
- [translated_history_context.py](src/translated_history_context.py)：保留连续均值图；按现Tab的−105编码直接作为特征均值，不阈值化或取整，真实S只加入一次。
- [translated_history_system_trial.py](src/translated_history_system_trial.py)：54事件、108次Tab前向已于11:39 UTC完整结束，572.20秒，无Tab训练或GP拟合。预先固定的主输出GP+读学习图的Tab为12.4767米，未改善原GP+ordinary系统10.7880米；同表示简单参考图+Tab组合11.8057米。所有预先声明的读出完整保留在[结果](evidence/translated_history_system_20261110/complete.json)与[冻结](evidence/translated_history_system_20261110/freeze.json)。
- `context_influence_*.py`补齐此前已保存的关系学习实现；v2正式训练仍暂停。旧接线检查对应此前版本，不能当作v2性能结果。

后续代码与关键结果继续同步本研究分支，保留双方提交；原始数据、大模型和凭据不上传。

### 已完成的整体推进

预先声明的另一条读出——学习翻译图的3NN与原GP固定等权——得到 **9.9607米**，较已有GP+ordinary系统10.7880米降低 **7.67%**。K1/3/5分别10.3638/9.8925/9.6258米，均值九层七层改善；同一3NN组合使用未学习参考图为12.8960米。总体位置均值误差p90为20.0522米（原20.7964），但K1为22.9095（原21.1161），少标定尾部仍需解决。

本轮采用原型的四层B0源资产，ordinary30仍是原三层受限源语料；结果是现有资产的系统集成，统一源语料的训练归因留待后续。保留历史翻译及有效GP互补，Tab维持原有原始历史用法；本次均值图替换输入未成功，不据此淘汰Tab或历史翻译机制。

已把有效算法整理为[纯H/S翻译入口](src/frozen_history_translator.py)和[可复用系统入口](src/translated_history_system.py)：加载已有五种子树模型，接收历史、现场标定与待定位扫描，复用原GP输出；不重新训练，不读取整份CSV或Query坐标。两个新入口经过静态审查，以上性能来自冻结的系统评价入口，未再启动数值重跑。原四组正式训练和关系编码器训练继续暂停。

### 真实扫描分布与直接定位

按用户最新要求，从现象追查学习对象，不把所有组件固定拼进最终方法。平均指纹会合并不同观测机制：50%未检出、检出时−65dBm，与始终−85dBm，按−105编码都得到均值−85；它们对实际扫描的定位证据不同。下一步沿历史翻译主线学习真实单次扫描的检出与RSS分布，统一形成位置证据。旧观测表与条件场已有相关思想，此处保留它们的经验，不另称首次提出概率定位。

[observation_translation_tasks.py](src/observation_translation_tasks.py)已将现43维历史＋12维标定表示接到原共同源任务：H∪S决定AP，只用H/S生成特征，源监督使用另一采集的真实逐扫描RSS并保留未检出值，位置/扫描/AP等权；不再以位置均值充当扫描标签。该模块可复用现有source_episode_catalog/episode_scans，保护源验证位置的权限沿原任务入口执行。对应的[共享概率模型](src/observation_translation_model.py)与[分开的源训练/目标评价入口](src/observation_translation_trial.py)已完成训练和54事件评价。模型把未检出与最多16个信号区间合为一个条件分布；区间只从合法源训练标签确定，S只用于条件特征。按AP独立的工作模型汇总扫描证据，在H/S已知位置上形成位置分布并采用既有距离风险决策，不叠加GP、Tab或3NN输出。源语料沿原full24/ordinary权限，限定B0F0–F2、72个跨采集训练任务和27个受保护源验证任务，目标评价单独启动。首轮源计算在拟合前因库对全NaN输入列分箱失败停止；已按源训练数据固定可观测列，保留部分缺失输入及原模型参数，失败记录保留。原四组训练与关系编码器训练继续暂停。已完成的9.9607米组合保留为实用结果与研究依据，最终结构不要求同时保留GP、Tab和3NN。

已全文读取[Claude框架v2](https://app.notion.com/p/414217e091ae81978a1cf668fdf2091a)。新增[faf_baseline.py](../vertical_translator_20261110/faf_baseline.py)按协作者原提交保留，未在本轮执行。逐点翻译误差方差没有定义跨位置协方差，且HS翻译已使用S；直接替换GP场并再次按S适应的方案尚不成立，不据页面建议启动这条接线或训练。恢复该接法需要明确的空间关系和一致的条件路径。当前优先完成真实观测监督，未启动轴向注意力网络、多数据集或新增下载。


已继续合入 Claude `6e40ef3`、`b721ca3`、`ae001b0`：保留似然代码的内存修改及物理楼层衰减、独立种子复核证据。B0-only固定楼层衰减方案H/HS为17.74/17.13米，五种子翻译器的已有HS为10.56米；另十个种子11–20的HS十模型集合约10.91米。这些是协作者现成结果，未重复运行。LOBO新种子部分仍未完成；此处不据单个楼层衰减方案概括全部物理方法。


### 真实观测学习的完整结果与下一步

[源学习](evidence/observation_translation_source_20261110_nanfix/complete.json)于140.66秒完成，72个合法源训练表拟合一次、27个受保护源验证事件平均5.5443米；[目标评价](evidence/observation_translation_target_20261110/complete.json)于392.61秒完成，54/54事件平均 **12.7186米**，位置均误差p90 **26.5510米**。同一批任务GP11.8983、ordinaryTab12.1148、原GP+ordinary系统10.7880米。新路径K1/3/5为12.5827/12.8267/12.7465米，增加标定尚未带来稳定改善；源端收益未迁移到整体目标表现。这一版本不替换已有9.9607米的四层源翻译组合，源模型与完整负结果保留。

进一步定位同一机制的限制：[observation_translation_readout_trial.py](src/observation_translation_readout_trial.py)固定已训练模型，用原72个源训练表的区间代表值将概率投影成图，同时将Query量化到相同代表值，比较固定3NN和原概率读出。候选/AP/S条件不变，不追加真实S指纹，不重训、不调温度、不加融合；这项事后读出比较已完成全部54事件，409.76秒，均值为12.6584米、p90为25.2486米；B1从10.9366退化为13.9023，B2从14.5007改善为11.4145。单独换读出没有解决整体迁移问题，不继续扫读出参数。完整结果见[读出比较](evidence/observation_translation_readout_20261110/complete.json)。

已继续合入Claude `44bebb7`、`69160df`、`b0800c6`，保留错误历史证据、观测分布hurdle候选与开发结果。其B0开发步骤训练于B1+B2，和本项目三层合法B0源→B1/B2目标的权限不同；现成读出参数不直接用于同信息比较，hurdle九层运行器本轮未运行。错误AP配对时30.24米接近S-only29.79米是协作者的该项结果，不作为任意错误历史下的无损保证。代码与证据保留双方提交历史。


沿同一翻译机制，代码审查确认原S表示会丢失部分多点空间对应：在历史参考恒定、最近点固定时，交换两个非最近点的RSS保持55维特征不变，而不同位置的联合条件更新可不同。这是表示的确定边界，尚未单独证明它导致上述目标误差。[joint_support_translation.py](src/joint_support_translation.py)已复用现有GP后验回调，保存完整S后验并在任意合法XY返回总先验/后验均值和边际方差，保留条件化产生的跨来源项；不相加各来源自身方差、不再次条件化S、不增加定位器。该接口已接到55维翻译context与原分类器，替换四项最近标定点特征，保留整体统计；源/目标状态需按原权限绑定。代码已静审，尚未计算或训练，作为可选能力保留，当前不激活。


### 接续连续观测分布的正向结果

Claude已完成连续hurdle翻译的九层评价：[B0种子1–5](../vertical_translator_20261110/evidence/hurdle_b0_ens5/summary.json)为**9.4325米**，[种子11–15](../vertical_translator_20261110/evidence/hurdle_b0_s11_15/summary.json)为**9.3664米**；[LOBO三种子](../vertical_translator_20261110/evidence/hurdle_lobo_ens3/summary.json)为9.5219米。这些现成代码和结果直接接续，未重复训练核验。B0版本使用四层全量源扫描，读出开发过程曾使用B1/B2训练场；与原三层合法源ordinary比较时分别标明信息与训练范围。页面尾部结果采用事件分位数的均值，与本项目总体位置均值误差p90口径不同。

新增[完整源扫描监督](src/continuous_observation_tasks.py)、[可部署连续翻译器](src/continuous_observation_translation.py)和[分开的学习/评价入口](src/continuous_observation_trial.py)。保持原H/S与72个源任务条件，把训练监督扩展为该任务Query采集中的全部合法源扫描；排除全局保护行及该draw完整K5标定坐标，原27项源验证不变。按位置保存检测次数、检测均值与波动，全部H/S选定AP单元均参与学习；检测、均值和跨楼层交叉拟合尺度共用原55维历史/标定表示。树拟合保持相对权重并固定均值1以匹配正则化尺度。部署仅接收H/S与Query RSS，沿Claude共享扫描偏移与固定读出直接形成定位，不追加GP融合或Tab分支。

新连续版本已完成[源学习](evidence/continuous_observation_source_20261110/complete.json)和[54事件目标评价](evidence/continuous_observation_target_20261110/complete.json)：2CPU单模型源训练及27项源验证共53.71秒，源均值5.1989米；固定模型目标评价322.05秒，平均 **10.0337米**、总体位置均误差p90 **20.1244米**。相比原GP11.8983米降低15.67%，相比GP＋ordinary系统10.7880米降低6.99%。K1/3/5依次10.4899/9.9185/9.6928米，三个预算均改善原系统均值，九层五层改善；K1尾部22.1328米仍弱于原21.1161米。四层原型GP＋翻译3NN的9.9607米保留为有效系统，未将其四层源与本版三层源差异抹平。

这轮推进建立了完整合法源监督下连续学习的实际作用，单路直接定位、不增加GP融合或Tab读出。模型及位置分布保存服务器，冻结配置和完整结果保留；开发读出参数曾涉及B1/B2来源，结果属于已曝光开发评价。9.43/9.37米仍是协作者原型的独立版本。用户随后明确以Claude最好完整版本为起点；本受限版不继续承担追分主线。联合标定context未训练，原四组正式Tab与关系编码器训练仍暂停。


## 原版五模型实际接入（用户最新取舍）

当前使用[Claude原版](src/claude_hurdle_20261110)，固定种子1–5与draws4，原训练代码逐文件保留。原发布脚本没有保存权重，经库存核实后仅补生成了可部署五模型，耗时245.06秒，共44.20 MB留服务器；首次运行在拟合前因遗漏已有pandas路径退出，复用原有依赖后成功，未安装或下载环境。实际解释器和依赖路径保存在[冻结记录](evidence/claude_hurdle_b0_ens5_20261110/freeze.json)。

[FrozenHurdleTranslator](src/frozen_hurdle_translator.py)加载这些权重，用显式H/S准备，再接收Query RSS；不依赖CSV全局状态，不改变原55维特征、分布集成、经验S或读出。一次B0源扫描的[部署对应检查](evidence/claude_hurdle_b0_ens5_20261110/deployment_check.json)与原程序位置、对数质量差均为0。没有重跑54个目标分数，该阶段最好已知9.3664米和种子1–5的9.4325米仍引用协作者已有结果，该阶段完成完整采用，尚未产生新的超越。

用户纠正前已完成的受限版[五标定位置记录](evidence/continuous_observation_support_stability_20261110/complete.json)和[尾部描述](evidence/continuous_observation_tail_20261110/audit.json)保留。五点K1平均9.9527米、各点9.6132–10.4899米，只描述该受限模型对既定标定位置的敏感性，不是原版提升，也不与单一旧标定点的对照冒充同信息比较。已停止在此版本上追加修正与实验。


### 原版历史翻译与GP校准的完整互补结果

[hurdle_gp_pool.py](src/hurdle_gp_pool.py)在原Hurdle候选行上读取已有GP概率面的双线性插值，固定等权对数结合；保留原Hurdle、原网格GP、候选限制后的GP、坐标half四项比较，不改训练配方、不挑权重。共享H/S和Query的两路是预测组合，未假设独立证据；平坦GP会温和化Hurdle，候选超出原网格则停止该接法。原54份GP二维分数缓存齐全，只需原五模型的新前向。完整评价入口[hurdle_gp_pool_trial.py](src/hurdle_gp_pool_trial.py)已完成54/54任务，184.95秒、2CPU、无新增训练或GP拟合。新增前向为新互补方法所需，原版单路仍为9.432476米。[完整结果](evidence/hurdle_gp_pool_20261110/complete.json)与[事前冻结](evidence/hurdle_gp_pool_20261110/freeze.json)保留全部比较。

合入协作者7149b46的新同口径p90及三层版本记录。原五种子Hurdle总体p90为18.02米、K1为17.70米；种子11–15总体16.82米、K1为19.45米。三层源协作者版本为9.59米单种子／9.79米五种子，仍与本项目受限权重/任务版本分别保留；不将差距直接归因于单个改动。


| 同批54个开发事件 | 平均误差（米） | 总体位置均误差p90（米） |
| --- | ---: | ---: |
| 原版Hurdle | 9.4325 | 18.0166 |
| 原GP | 11.8983 | 24.1167 |
| 候选位置上的GP | 10.5009 | 21.2356 |
| 原两路坐标half | 9.9508 | 20.7601 |
| **固定概率half（采用）** | **8.7792** | **15.9539** |

K1/3/5均值8.9516/8.7316/8.6545米，K1 p90为15.7775米（原版17.6988）。两栋建筑平均均改善，九楼八楼平均改善；B1F1均值8.949→8.965、B1整体p90 13.441→13.637略退，完整保留。该结果支持两路位置证据的实际互补，两个GP版本均完整报告，候选限制带来的收益与两路互补的新增收益分别可见；该组合使用相关的预测证据。原GP在新部署情境中的推断成本仍需承担，本次184.95秒使用其已有缓存。当前成果属于已曝光开发任务，下一阶段保留这一完整方法做确认，不追加权重扫描或改回受限重写版。


### IPIN2025 固定方法外部确认：训练历史已完成，正式评分尚未运行

原五模型与固定概率half继续使用。新增[训练资料接口](src/ipin2025_observation_data.py)、[共有历史准备](src/ipin2025_history_assets.py)、[上下文准备及因果预测入口](src/ipin2025_confirmation.py)和[独立评分器](src/ipin2025_score.py)。以往公开轨迹缓存的坐标FPS预算本次前瞻性用于IPIN2025：双方共用每层最多128个历史位置、每位置一条实际receipt；F6只有60个可用位置。K1/3/5的标定严格嵌套，AP轴只由选定H与当前K的S决定。源模型和GP全局参数不重训，保持原GP完整后验和适应路径。

[历史资产结果](evidence/ipin2025_history_assets_20261110/complete.json)已完成1800个AP场，94.94秒CPU2、无下载，最大投影梯度2.02e-6；[冻结记录](evidence/ipin2025_history_assets_20261110/freeze.json)保留输入、配方及源模型身份。模型和NPZ仍留服务器。现场H/S上下文准备与正式预测是独立阶段，后续实际运行状态以对应结果为准；新增predict/score入口尚未执行，无外部定位分数。

只用官方Training资料准备H/S；所有合格楼层2–6×K1/3/5×两日志共30流先封存，再按GT楼层选择预测。原AppTimestamp的绝对2Hz时钟，使用已收到的最近WiFi；评分为最后不晚于GT+0.01秒的预测。Level1缺少训练覆盖、无先前预测的点留在全场覆盖分母；不删除陈旧快照。评价的是已知楼层条件下的二维定位，不是官方跨层竞赛成绩。训练坐标保留POSI时间插值来源。

本提交同时保留并合入Claude9432a5c及其新增源端选择证据：B0楼内留一层仍选β128、ε.02、偏移积分（5.93米）。这是九层开发结果之后的源端佐证；此前九层身份保持开发评价。IPIN正式预测/评分前配置已固定，不追溯改写实验时间顺序。


IPIN2025首个H/S部署上下文现已实际完成（F2/K1）：1009个AP、4864个GP网格点、445个Hurdle候选，CPU2准备172.74秒，其中适应1.88秒；最低ESS31,122.6。原 `--max-contexts 1` 正常结束为退出码2。一次唯一Training Support receipt读取合计0.3211秒（学习端0.0077、GP0.3123、融合0.0011），只核实部署前向与单次成本，未计算定位误差。完整流吞吐量以正式运行计时为准。[首上下文记录](evidence/ipin2025_confirmation_20261110/first_context.json)及[训练侧成本](evidence/ipin2025_confirmation_20261110/training_readout_cost.json)保留实际状态；其余上下文继续同一冻结方案分段准备，正式Query/GT和评分尚未开启。
