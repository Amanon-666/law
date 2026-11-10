# 楼层翻译器（2026-11-10，Claude）

机制说明、全部结果和边界见 Notion：[突破｜楼层翻译器](https://app.notion.com/p/414217e091ae81a49454ecc514f7bcda)。本目录是可运行的代码和轻量证据。

## 是什么

一个在其他楼上训练的回归器。对目标层的每个候选位置（即其他楼层的历史位置）和每个 AP，它根据两类输入预测该 AP 在目标层这个位置的信号强度：

- 该 AP 在其他各层同一平面位置的强度，按楼层差排列；
- 少量标定 S 相对历史参照的偏差。

输入里没有 AP 编号，也没有绝对坐标。训练时让训练楼的每一层轮流当目标层，标签是这一层真实的位置均值图。部署时只读新楼的 H 和 S。翻译出来的图交给读图器，读图器目前是项目的 3NN。

## 运行

只依赖 numpy、pandas、scikit-learn，CPU 上几分钟。必须设置数据路径：

    export LAW_UJI_CSV=/path/to/UJIndoorLoc/trainingData.csv   # SHA256 45ca0128...5168
    python run_nine_floor.py --train lobo   --seeds 1 2 3 4 5 --out maps_lobo_ens5
    python run_nine_floor.py --train b0only --seeds 1 2 3 4 5 --out maps_b0_ens5

验收：`--train b0only --seeds 1` 应输出 translated_H 10.991、translated_HS 10.841。`core.py` 的 3NN(H+S) 复现项目值 16.8254。

每个事件写出 `<out>/<event_id>.npz`，字段如下：

- `cand_xy`：候选位置坐标
- `map_H`、`map_HS`：翻译后的目标层编码 RSS，单位 dBm，−105 表示未检出
- `aps`：项目 AP 规则选出的 AP
- `pred_3nn_*`：3NN 读图给出的坐标
- `query_rows`、`bounds`

`tab_adapter.py` 把一张图转成 `FrozenHistoryRegressor.fit` 已经接受的 `dict(rss, xy)` 格式。

## 文件

| 文件 | 内容 |
|---|---|
| `protocol9.py` | `current_20261110/src/building_information_protocol.py` 的字节拷贝 |
| `core.py` | 数据、项目清单、3NN、计分与聚合、B0 开发折 |
| `translator.py` | 历史特征（按楼层差排列）、标定特征 |
| `vt_lib.py` | 地图、训练集构造、模型、`translate`、3NN 读图 |
| `run_nine_floor.py` | 训练、九层评价、导出翻译图 |
| `tab_adapter.py` | 翻译图转成 Tab 上下文格式 |
| `likelihood.py` | 异方差翻译加整幅扫描似然读图器：把 GP 路径的强项（似然、逐格不确定度、后验均值）放进学习得到的翻译里 |
| `dev_likelihood.py` | 只在 B0 开发折上选定似然读图器的两个参数 |
| `wrong_history.py` | 反例检验：历史 AP 列被打乱时的表现 |
| `evidence/` | 各轮日志和 JSON |
| `dev_scripts/` | 开发过程脚本，原样保留作记录；导入路径是开发机路径，不能直接运行 |

## 结果摘要（九层 54 事件，3NN 读图）

项目参照：

| 方法 | 平均误差 (m) |
|---|---|
| 3NN | 16.83 |
| GP | 11.90 |
| 普通微调 Tab | 12.11 |
| GP+Tab 等权平均 | 10.79 |

翻译器：

| 设置 | 只看 H | 看 H+S |
|---|---|---|
| 两栋源楼（LOBO），种子 1–5 的图取平均 | 11.55 | 10.43 |
| 仅 B0，种子 1–5 的图取平均 | 11.39 | 10.57 |
| 仅 B0，复核种子 11–20 的图取平均 | 11.63 | 10.91 |
| 仅 B0，种子 11–20 单个种子 | 11.45 ± 0.26 | 11.13 ± 0.49 |

只看 H 的翻译在各设置下都稳定，好于 GP 和所有 Tab。用 S 修正的效果依赖源楼数量。

这不是盲测：B1、B2 是已经暴露的开发楼。
