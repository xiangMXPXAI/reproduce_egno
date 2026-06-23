# EGNO 数据集构建与输入输出说明

本文档面向本项目的复现实验，结合论文 *Equivariant Graph Neural Operator for Modeling 3D Dynamics* 与仓库中的数据读取代码，说明四类数据集的来源、文件结构、预处理方式、模型输入输出和任务目标。

## 1. 统一任务形式

EGNO 将不同物理系统统一为 3D 图动力学预测问题。每个样本可抽象为：

```text
输入：G(t0) = {x0, v0, h0, E, e}
输出：{x(t1), x(t2), ..., x(tP)}
```

其中：

| 符号 | 含义 |
|---|---|
| `x0` | 初始时刻节点的 3D 坐标 |
| `v0` | 初始速度，由真实轨迹差分得到或仿真直接提供 |
| `h0` | 节点标量特征，例如速度模长、原子类型、电荷等 |
| `E` | 图边索引 |
| `e` | 边特征，例如电荷乘积、hop type、距离、接触距离 |
| `P` | 一次前向预测的未来离散时间点数量 |

训练和测试中的核心指标遵循论文定义：

- `F-MSE`：Final MSE，仅计算最后一个预测时间点的位置均方误差。
- `A-MSE`：Average MSE，计算所有预测时间点的位置均方误差平均。

代码中的训练日志 `avg loss` 与 `F-MSE` 对齐：训练目标是所有时间步 MSE 的均值，但日志统计的是最后一步误差。

## 2. N-body Simulation

相关文件：

- `simulation/dataset/generate_dataset.py`
- `simulation/dataset/synthetic_sim.py`
- `simulation/dataset_simple.py`
- `main_simulation_simple_no.py`
- `configs/config_simulation_simple_no.json`

### 2.1 物理背景

N-body 数据集模拟 5 个带电粒子在三维空间中的相互作用。粒子运动由类似库仑力的相互作用驱动：

```text
F_ij ∝ c_i c_j / ||x_i - x_j||^3
```

其中 `c_i` 是粒子电荷。该任务要求模型根据初始位置、速度和电荷关系，预测未来多个时间点的粒子位置。

### 2.2 数据生成

数据由 `ChargedParticlesSim` 生成。默认参数包括：

| 参数 | 含义 |
|---|---|
| `n_balls=5` | 粒子数量 |
| `dim=3` | 三维空间 |
| `vel_norm=0.5` | 初始速度归一化尺度 |
| `noise_var=0.0` | 默认无观测噪声 |
| `sample_freq=100` | 仿真中每隔若干积分步保存一帧 |
| `length=5000` | 每条轨迹积分长度 |

推荐生成方式：

```bash
cd simulation/dataset
mkdir -p simple
cd simple
python ../generate_dataset.py --num-train 10000 --num-valid 2000 --num-test 2000 --seed 43 --sufix small
```

`dataset_simple.py` 默认读取后缀 `_charged5_initvel1small`，因此需要使用 `--sufix small`。

### 2.3 文件内容

生成后的文件位于：

```text
simulation/dataset/simple/
```

主要文件：

| 文件 | 原始形状 | 含义 |
|---|---:|---|
| `loc_train_charged5_initvel1small.npy` | `[S, T, 3, N]` | 粒子位置轨迹 |
| `vel_train_charged5_initvel1small.npy` | `[S, T, 3, N]` | 粒子速度轨迹 |
| `edges_train_charged5_initvel1small.npy` | `[S, N, N]` | 电荷乘积 `c_i c_j` |
| `charges_train_charged5_initvel1small.npy` | `[S, N, 1]` | 粒子电荷 |

加载时，`NBodyDataset.preprocess()` 将 `loc/vel` 转换为：

```text
[S, T, N, 3]
```

并构建无自环有向全连接图。对于 `N=5`，每个样本有 `5 × 4 = 20` 条有向边。

### 2.4 单样本输入输出

`NBodyDynamicsDataset.__getitem__()` 返回：

```python
loc[frame_0], vel[frame_0], edge_attr, charges, locs
```

当前 `nbody_small` 设置：

```text
frame_0 = 30
frame_T = 40
P = 5
目标帧 = 32, 34, 36, 38, 40
```

输入：

- `loc[frame_0]`: `[5, 3]`
- `vel[frame_0]`: `[5, 3]`
- `edge_attr`: `[20, 1]`，边上的 `c_i c_j`
- `charges`: `[5, 1]`

训练时额外拼接：

```python
loc_dist = ||x_i - x_j||^2
edge_attr = concat([c_i c_j, loc_dist])
```

节点特征：

```python
h_i = ||v_i||_2
```

输出：

- `locs`: `[5, P, 3]`
- 模型内部展平为 `[P * B * N, 3]`

## 3. Motion Capture

相关文件：

- `motion/dataset.py`
- `motion/dataset/motion.pkl`
- `motion/dataset/motion_run.pkl`
- `main_mocap_no.py`
- `configs/config_mocap_no.json`

### 3.1 数据背景

Motion Capture 数据集来自 CMU 动作捕捉数据。项目复现两个动作：

| case | 含义 |
|---|---|
| `walk` | Subject #35 Walk |
| `run` | Subject #9 Run |

人体骨架被建模为图：

- 节点：31 个关节点
- 边：人体骨架连接及其二跳邻居
- 坐标：每个关节点的 3D 位置

任务目标是根据人体初始姿态和速度预测未来人体运动。

### 3.2 文件内容

预处理文件位于：

```text
motion/dataset/
```

| 文件 | 内容 |
|---|---|
| `motion.pkl` | Walk 动作，保存 `(edges, X)` |
| `motion_run.pkl` | Run 动作，保存 `(edges, X)` |
| `split.pkl` | Walk 的 train/val/test split |
| `split_run.pkl` | Run 的 train/val/test split |

`X` 是动作序列列表，每条序列形状为：

```text
[frames, 31, 3]
```

速度由代码差分得到：

```python
V[t] = X[t + 1] - X[t]
```

### 3.3 数据划分

代码中手动指定不同动作序列的划分：

- Walk：训练、验证、测试分别从固定 case id 中采样。
- Run：同样按固定 case id 划分。

当前配置：

| 实验 | 训练样本 | 验证样本 | 测试样本 | `delta_frame` | `P` |
|---|---:|---:|---:|---:|---:|
| Walk | 500 | 600 | 600 | 30 | 5 |
| Run | 200 | 240 | 240 | 30 | 5 |

### 3.4 图结构与输入输出

`MotionDynamicsDataset` 构建：

| 边类型 | 边特征 |
|---|---|
| 一跳骨架边 | `1` |
| 二跳邻居边 | `2` |

返回值：

```python
x_0, v_0, edges, edge_attr, local_edges, local_edge_attr, node_fea, x_t, v_t
```

输入：

- `x_0`: `[31, 3]`
- `v_0`: `[31, 3]`
- `edges`: `[2, E]`
- `edge_attr`: `[E, 1]`
- `node_fea`: `[31, 1]`，代码使用 `y` 坐标归一化特征

训练时节点特征拼接：

```python
h_i = [||v_i||_2, node_fea_i]
```

边特征拼接：

```python
edge_attr = [hop_type, ||x_i - x_j||^2]
```

输出：

- `x_t`: `[31, P, 3]`
- `v_t`: `[31, P, 3]`

代码中的 `MotionDynamicsDataset` 使用最后连续 `P` 帧作为预测目标：

```text
delta_frame = 30, P = 5
目标相对帧 = 26, 27, 28, 29, 30
```

## 4. MD17

相关文件：

- `md17/dataset.py`
- `main_md17_no.py`
- `configs/config_md17_no.json`

### 4.1 数据背景

MD17 是小分子分子动力学数据集，包含多个分子的长时间原子坐标轨迹。项目当前复现实验使用：

```text
mol = aspirin
```

任务目标是根据小分子当前构象和速度预测未来构象。

### 4.2 文件内容

原始数据放置在：

```text
md17/md17_<molecule>.npz
```

例如：

```text
md17/md17_aspirin.npz
```

代码读取：

| 字段 | 含义 |
|---|---|
| `R` | 原子坐标轨迹，形状通常为 `[frames, atoms, 3]` |
| `z` | 原子序数 |

项目提供 split 文件：

```text
aspirin_split.pkl
benzene_old_split.pkl
...
```

训练时通过 `max_training_samples=500` 只取前 500 个训练样本，验证/测试各取 2000 个。

### 4.3 预处理逻辑

代码首先计算速度：

```python
v[t] = R[t + 1] - R[t]
```

然后移除氢原子：

```python
z > 1
```

因此模型只在 heavy atoms 上预测。

当前 Aspirin 原始原子序数为：

```text
[6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]
```

移除氢原子后，保留 13 个 heavy atoms。

### 4.4 分子图构建

`MD17DynamicsDataset` 根据初始帧距离构建分子图：

```python
if distance(i, j) < 1.6:
    atom_edges[i][j] = 1
```

随后加入二跳邻居：

```python
atom_edges2 = atom_edges @ atom_edges
```

边特征包括：

```text
[atom_type_i, atom_type_j, hop_type, stick_ind]
```

其中 `stick_ind` 来自 `sample_cfg()` 中手工定义的刚性约束边。

训练/评估时额外拼接当前几何距离平方：

```text
[atom_type_i, atom_type_j, hop_type, stick_ind, ||x_i - x_j||^2]
```

因此 MD17 模型配置为：

```python
in_edge_nf = 5
```

### 4.5 输入输出

返回值：

```python
x_0, v_0, edge_attr, mole_idx, x_t, v_t, Z, cfg
```

输入：

- `x_0`: `[N, 3]`
- `v_0`: `[N, 3]`
- `edge_attr`: `[E, 4]`
- `mole_idx`: `[N, 1]`
- `Z`: `[N, 1]`

节点特征：

```python
h_i = [||v_i||_2, Z_i / max(Z)]
```

输出：

- `x_t`: `[N, P, 3]`
- `v_t`: `[N, P, 3]`

当前复现实验配置：

```text
delta_frame = 3000
P = 8
目标帧 = st + 3000 * i / 8, i = 1,...,8
```

## 5. Protein MDAnalysis

相关文件：

- `mdanalysis/preprocess.py`
- `mdanalysis/dataset.py`
- `main_mdanalysis_no.py`
- `configs/config_mdanalysis_no.json`

### 5.1 数据背景

Protein 数据使用 MDAnalysisData 中的 ADK equilibrium trajectory，即腺苷酸激酶 apo adenylate kinase 的分子动力学轨迹。项目默认只使用 protein backbone 原子。

任务目标是根据当前 backbone 构象、速度和接触图预测未来构象。

### 5.2 预处理缓存

运行：

```bash
python mdanalysis/preprocess.py --dir mdanalysis/dataset
```

会生成：

```text
mdanalysis/dataset/adk_backbone_processed/
```

主要文件：

| 文件 | 内容 |
|---|---|
| `adk.pkl` | 静态化学键、键长、电荷、帧数 |
| `adk_0.pkl`, `adk_1.pkl`, ... | 每帧位置、速度、空间接触图、接触距离 |

每个逐帧缓存包含：

```python
loc, vel, edges_global, edges_global_attr
```

### 5.3 图结构

Protein 数据同时使用两类图信息：

| 图 | 来源 | 作用 |
|---|---|---|
| local graph | 拓扑化学键 | 表示稳定局部连接 |
| global graph | cutoff 接触图 | 表示当前空间邻近关系 |

当前评估模型使用 `edge_global` 作为消息传递图，边特征为接触距离，并额外拼接当前距离平方。

注意：论文附录描述 Protein cutoff 为 `10 Å`，但当前代码 `preprocess.py` 中默认：

```python
cut_off = 8
```

本复现保持代码默认设置。

### 5.4 输入输出

`MDDynamicsDataset` 返回：

```python
loc_0, vel_0, edge_global, edge_global_attr, edges, edge_attr, charges, loc_t, vel_t
```

经 `collate_mdd()` 后输入模型：

- `loc_0`: `[B, N, 3]`
- `vel_0`: `[B*N, 3]`
- `edge_global`: `[2, E_batch]`
- `edge_global_attr`: `[E_batch, 1]`
- `charges`: `[B*N, 1]`

节点特征：

```python
h_i = [||v_i||_2, charge_i / max(charge)]
```

输出：

- `loc_t`: `[P * B * N, 3]`
- 可还原为 `[P, B, N, 3]`

当前配置：

```text
delta_frame = 15
P = 4
目标相对帧 = 3, 7, 11, 15
```

## 6. 数据集对比总览

| 数据集 | 节点 | 边 | 节点特征 | 边特征 | 输出 |
|---|---|---|---|---|---|
| N-body | 带电粒子 | 全连接无自环 | 速度模长 | 电荷乘积、距离平方 | 未来粒子位置 |
| Motion | 人体关节 | 骨架边、二跳边 | 速度模长、坐标特征 | hop type、距离平方 | 未来人体姿态 |
| MD17 | heavy atoms | 一跳/二跳分子图 | 速度模长、原子类型 | 原子类型、hop、约束、距离平方 | 未来分子构象 |
| Protein | backbone atoms | 空间接触图 | 速度模长、电荷 | 接触距离、距离平方 | 未来蛋白 backbone 构象 |

这些数据集覆盖了从低维粒子系统到人体运动、分子动力学和蛋白质构象预测的多尺度 3D 动力学任务，是检验 EGNO 是否能同时建模几何结构和时间相关性的核心 benchmark。
