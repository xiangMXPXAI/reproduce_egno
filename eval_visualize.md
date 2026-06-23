# EGNO 评估与可视化说明

本文档说明 `eval_visualize_*.py` 脚本的评估口径、输入输出含义、参数设置和绘图内容。

## 1. 论文指标

EGNO 论文对 state-to-state 和 state-to-trajectory 使用两类核心指标：

- `F-MSE`：Final Mean Squared Error，只计算最后一个预测时间点的位置均方误差。
- `A-MSE`：Average Mean Squared Error，计算所有离散预测时间点的位置均方误差平均。

代码同时输出：

- `F-MSE_x1e2` / `A-MSE_x1e2`：乘以 `100` 后的指标，便于和论文中 `MSE (x10^-2)` 表格对照。
- `F-MAE` / `A-MAE`：绝对误差辅助观察。
- `mean_final_node_l2`：最后一帧逐节点欧氏距离误差平均值。

训练代码中的日志 `avg loss` 与论文 `F-MSE` 对齐：它统计最后一个预测时间点的 MSE。

## 2. 通用运行方式

在 `egno` 根目录执行：

```bash
python egno_eval_visualize.py
```

或分别运行：

```bash
python eval_visualize_simulation.py
python eval_visualize_mocap.py
python eval_visualize_md17.py
python eval_visualize_mdanalysis.py
```

常用参数：

| 参数 | 含义 |
|---|---|
| `--logs-dir` | 权重与配置所在目录，默认 `logs` |
| `--output-dir` | 图片和指标输出目录 |
| `--experiments` | 指定实验目录名，例如 `simulation_exp` |
| `--device` | `auto` / `cpu` / `cuda` |
| `--batch-size` | 评估 batch size，`0` 表示使用 `config.json` 中的值 |
| `--max-batches` | 只评估前若干个 batch，`0` 表示完整测试集 |
| `--num-visual-samples` | 为多少个测试样本单独绘图并记录单样本指标 |
| `--dpi` | 图片分辨率 |
| `--max-plot-nodes` | 图中最多绘制多少个节点，蛋白质等大图会自动抽样 |
| `--max-plot-edges` | 图中最多绘制多少条边 |

快速检查示例：

```bash
python eval_visualize_md17.py --max-batches 2 --num-visual-samples 3
```

完整测试示例：

```bash
python eval_visualize_mocap.py --device cuda
```

## 3. 输出文件

每个脚本会在对应 `outputs/` 子目录生成：

```text
metrics.json
dataset_overview.png
training_curves.png
<exp_name>_prediction.png
<exp_name>_samples/
    sample_metrics.json
    <exp_name>_sample_000.png
    <exp_name>_sample_001.png
    ...
```

其中：

- `metrics.json`：完整测试集指标，包含论文口径 `F-MSE` 和 `A-MSE`。
- `sample_metrics.json`：若干单样本指标。
- `*_prediction.png`：该实验第一个可视化样本的综合图。
- `*_sample_*.png`：单样本预测图。
- `training_curves.png`：训练、验证、测试 loss 曲线。
- `dataset_overview.png`：数据集输入输出和目标说明总览。

## 4. 各数据集输入、输出与目标

### N-body Simulation

脚本：

```bash
python eval_visualize_simulation.py
```

输入：

- `loc[frame_0]`：5 个带电粒子的初始 3D 坐标。
- `vel[frame_0]`：初始速度。
- `edge_attr`：粒子两两电荷乘积 `c_i c_j`。
- 评估时额外拼接当前几何距离 `||x_i - x_j||^2`。

输出/真实值：

- 未来 `P=5` 个离散时间点的粒子 3D 坐标。
- `F-MSE` 比较最后一个时间点预测位置和真实位置。
- `A-MSE` 比较整个未来轨迹。

目标：

- 学习带电粒子系统中的库仑相互作用动力学算子。

### Motion Capture

脚本：

```bash
python eval_visualize_mocap.py
```

输入：

- 31 个人体关节点初始 3D 坐标。
- 31 个人体关节点初始速度。
- 骨架边和二跳边。
- 节点特征为代码中的归一化坐标特征，与训练脚本保持一致。

输出/真实值：

- 未来 `P=5` 个时间点的人体骨架姿态。
- Walk 和 Run 分别对应 `mocap_exp_walk` 与 `mocap_exp_run`。

目标：

- 根据人体骨架初态预测后续运动轨迹。

### MD17

脚本：

```bash
python eval_visualize_md17.py
```

输入：

- 小分子 heavy atoms 初始 3D 坐标。
- heavy atoms 初始速度。
- 原子序数作为节点类型。
- 一跳/二跳分子图边特征，包括原子类型、hop type 和约束标记。
- 评估时额外拼接当前原子间距离平方。

输出/真实值：

- 未来 `P=8` 个时间点的小分子构象。
- 当前配置中的 `delta_frame=3000`，即最后一个预测快照距离输入状态 3000 帧。

目标：

- 学习小分子分子动力学轨迹预测算子。

### Protein MDAnalysis

脚本：

```bash
python eval_visualize_mdanalysis.py
```

输入：

- ADK 蛋白 backbone 原子初始 3D 坐标。
- backbone 原子初始速度。
- 空间接触图 `edge_global`。
- 接触距离作为边特征。
- 电荷/原子属性作为节点特征来源。

输出/真实值：

- 未来 `P=4` 个时间点的蛋白 backbone 构象。
- 当前配置中 `delta_frame=15`。

目标：

- 预测蛋白质 backbone 在分子动力学轨迹中的构象演化。

## 5. 绘图内容说明

每张预测图包含：

1. 左侧说明卡片：数据集、实验名、输入、输出、任务目标、`F-MSE/A-MSE`。
2. `Input: initial state`：初始坐标、图边和速度方向。
3. `Output: future trajectory`：真实轨迹与预测轨迹对比。
4. `Final-step spatial error`：最后一个时间点逐节点误差热力图。
5. `Temporal error profile`：每个预测时间点的 MSE 和 MAE。
6. `Prediction displacement vectors`：最后一帧预测点到真实点的误差连线。

颜色约定：

- 真实轨迹：浅色实线。
- 预测轨迹：数据集主题色虚线。
- 误差：`magma` 热力色，颜色越亮误差越大。

