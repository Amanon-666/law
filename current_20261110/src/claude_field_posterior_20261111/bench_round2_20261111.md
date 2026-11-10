# 第二轮基准执行事实（2026-11-11）

截至UTC2026-11-10 22:04:55（北京时间2026-11-11 06:04:55），冒烟与UJI三折全部完成，正式195任务完成、0失败。Tampere两栋未启动。当前无运行中的实验PID。

## 冒烟

B1/F0/K1/r0，runner3694177、Python3694181；UTC21:09:39—21:21:13，694秒，exit0。源拟合与预测均completed，仅验证链路，不计入正式三折。

|method|mean|p50|p90|K1 mean|K3 mean|K5 mean|fit seconds (sum)|task seconds (mean)|completed|failed/unavailable|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ours_A|11.881998369698769|11.301508992741738|20.912569448307668|11.881998369698769|—|—|690.6747305691242|3.128619506955147|1|0|

## 正式UJI各折

### bench_uji_B0_A_20261111

|method|mean|p50|p90|K1 mean|K3 mean|K5 mean|fit seconds (sum)|task seconds (mean)|completed|failed/unavailable|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ours_A|5.3440490004061045|4.300382197545378|10.525898146696473|5.481302894448561|5.253195341235267|5.297648765534484|624.4495405070484|2.289080976632734|60|0|

原日志：`start=2026-11-10T21:22:21Z runner_pid=3711710`；`child_pid=3711713`；`end=2026-11-10T21:35:03Z elapsed_seconds=762 exit_code=0`。

### bench_uji_B1_A_20261111

|method|mean|p50|p90|K1 mean|K3 mean|K5 mean|fit seconds (sum)|task seconds (mean)|completed|failed/unavailable|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ours_A|11.516520919514667|9.108318650213182|22.160865253267517|11.838216013028093|11.520662899969674|11.190683845546234|718.431787583977|3.2224627073233325|60|0|

原日志：`start=2026-11-10T21:36:27Z runner_pid=3731515`；`child_pid=3731519`；`end=2026-11-10T21:51:39Z elapsed_seconds=912 exit_code=0`。

### bench_uji_B2_A_20261111

|method|mean|p50|p90|K1 mean|K3 mean|K5 mean|fit seconds (sum)|task seconds (mean)|completed|failed/unavailable|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ours_A|9.770460616358871|6.706287340467904|20.987708086075802|9.984028282433739|9.85741432292951|9.469939243713362|404.14194589853287|4.7440137445926664|75|0|

原日志：`start=2026-11-10T21:52:02Z runner_pid=3753688`；`child_pid=3753692`；`end=2026-11-10T22:04:43Z elapsed_seconds=761 exit_code=0`。

## UJI三折合并

|method|mean|p50|p90|K1 mean|K3 mean|K5 mean|fit seconds (sum)|task seconds (mean)|completed|failed/unavailable|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ours_A|9.034980835708025|6.351877610616509|18.4505958641021|9.258112959298463|9.048784870512112|8.7980446773135|1747.0232739895582|3.5204802660605847|195|0|

口径与score.aggregate一致：先用每任务记录的position_errors，再拼接全部成功位置计算mean/p50/p90；K分组同理。不是三个楼栋分数简单平均。全局p50/p90也不是各任务分位数平均。source_fit秒数包括runner保存模型的时间，合并行取三折之和；每任务秒数表内为事件seconds平均，全部逐任务原值保存在events JSON。本轮各折冻结的law_bench代码哈希一致。

## Tampere库存与未运行项

实际调用law_bench.data.load('tampere')确认如下；使用原训练数据，无下载。

|建筑|层|位置数|扫描数|检测AP数|建筑AP列数|
|---|---:|---:|---:|---:|---:|
|TIE1|-1|171|171|114|613|
|TIE1|0|5972|5972|519|613|
|TIE1|1|2475|2475|332|613|
|TIE1|2|1007|1007|188|613|
|TIE1|3|930|930|119|613|
|TIE1|4|78|78|19|613|
|SAH1|1|1951|1951|380|775|
|SAH1|2|5293|5293|643|775|
|SAH1|3|2047|2047|376|775|

TIE1六层，SAH1三层；每个精确XY仅一条扫描，R=3上限不会制造重复扫描。两栋所有指定方法均未启动，完成/失败数均为0（不表示成功，分母尚未执行），因此没有mean/p50/p90、K分数、fit或任务耗时。

指定选项仅设置gp_histmean.readout=map；retained_gp构造另需parameters_path、provenance、cache_dir。静态核查表明原命令将因缺参失败；这是预计失败而非已运行失败。未擅自补配置。adopted_A在Tampere是否成功尚未运行核验。

## 版本、审查与修正清单

- 本机macOS，服务器ray-web/Linux；仅操作指定独立worktree `/home/panyushuo/projects/panyushuo/claude_fp_wt`，分支`research/claude-field-posterior-20261111`。
- 运行前HEAD `87c96d1`。fetch后current最新`55cf111`，merge返回Already up to date，无冲突。后续fetch发现current新增`713c001`，未并入已经开始的冻结运行。
- 四个指定Claude文件通读并原样接入law_bench，本机/服务器SHA逐一一致。源码bug修正清单：无。只新增用户指定的run_claude进程内注册入口，不手动修改run.py/protocol.py或其他非指定模块。
- 源拟合只用允许源建筑，UJI各折源为另两栋。目标H为其他楼层，S进入支持特征及经验候选，Q仅RSS进入预测；Q坐标只交评分器，AP轴只由H/S选择。A的源支持特征和监督可能来自相同源位置，不等于目标Q泄漏，也不等于独立源验证。
- A使用支持条件算子及scale读出，不走M2。输出受H/S候选范围限制；错误历史可能造成偏差，新AP只有H/S发现才入轴。空S支持特征返回NaN（本协议K≥1）；单锚点信息有限。条件化分支零残差不移动均值，高噪声削弱校正；距离核共同平移不变，unique聚合不依赖行序。这些为事前代码逻辑检查，未扩展性能实验或改变原方法。
- 复用指定解释器和冻结PYTHONPATH、VT_DIR、LAW_UJI_CSV。run.main设置nice+10，外层不叠加；四项线程2、GPU禁用、离线标志和PYTHONDONTWRITEBYTECODE开启。每次nohup串行运行，无模型/数据下载、无依赖安装、无本机实验。
- 本机首次拉回路径漏掉claude_handoff，已移至正确目录，仅清理自行创建的空目录。汇总时一次默认python缺numpy，改用已有conda解释器读取JSON，未安装依赖。

四文件SHA256：ours `1e9a526126cf5a9580aeac696aeef3435048daf5827630a858779f697e5688d8`；methods_claude `0ea96b65e6fa8622808d0978e5215477fe0c1ca0dba5f44ce1ca00735f1e784b`；gp_lit `26e158ecac34ba2b617682798b803f906324c263ff09b942cb97f80473b32ef7`；run_claude `5ba3962353556e55ae91f74ca7f758601736e3f18a7ef6fc1e001ea013b9bdfc`。

## 提交与时间界限

|内容|已推送提交|
|---|---|
|源码、runner/选项、冒烟及Tampere库存|eebe366|
|UJI B0轻量结果|acd89c8|
|UJI B1轻量结果|922e00d|
|UJI B2轻量结果|45414d3|

每次提交前fetch，本研究分支无协作者新提交；源码及轻量JSON/日志同步，模型留服务器、不进Git。各运行完成后立即提交推送。

从冒烟启动计55分钟界限为UTC22:04:39；B2在22:04:43自然完成，晚4秒。22:04:55最终检查runner3753688/child3753692已不存在；没有杀进程，也没有启动后继Tampere。此前22:04:26该child仍计算中，CPU193%、RSS856920KiB。后续工作仍为指定TIE1→SAH1串行执行并补表，不需要重跑UJI。

启动、冒烟、B0/B1关键回执已写入现有LAW通信讨论；Notion成功不表示Claude已读或另行转达成功。未修改bridge队列/程序、主线程worktree、冻结依赖或他人进程。

Git经既有mac-net，初次fetch代理payload to_mac=1230/from_mac=5582字节；不含探测/SSH/TLS/scp，不能代表校园网计费量（未知）。后续传输无统一完整字节总计；只传源码与轻量结果，无数据/模型下载。
