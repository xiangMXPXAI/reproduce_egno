# EGNO 算法模型与代码实现说明

本文档结合论文 *Equivariant Graph Neural Operator for Modeling 3D Dynamics* 与本项目代码，说明 EGNO 的问题建模、核心网络结构、前向传播流程、训练配置和复现中新增的评估可视化模块。

## 1. 方法定位

EGNO 的目标是学习 3D 图动力系统的算子映射：

```text
F_theta: G(t0) -> {G(t1), G(t2), ..., G(tP)}
```

与普通 GNN 单步预测不同，EGNO 一次前向传播直接输出未来多个时间点的几何状态。它结合两类结构：

1. **EGNN 空间消息传递**：保证 3D 平移、旋转等几何变换下的等变性。
2. **Temporal Neural Operator / Fourier temporal convolution**：显式建模未来轨迹不同时间点之间的相关性。

因此，EGNO 可以被理解为：

```text
EGNO = EGNN spatial equivariance + neural operator temporal modeling
```

## 2. 代码结构

核心模型文件：

```text
model/
├── basic.py      # EGNN、等变/不变标量网络、基础 MLP、聚合函数
├── layer_no.py   # 时间嵌入、SpectralConv、TimeConv、TimeConv_x
└── egno.py       # EGNO 主模型
```

训练入口：

```text
main_simulation_simple_no.py
main_mocap_no.py
main_md17_no.py
main_mdanalysis_no.py
```

复现实验辅助模块：

```text
training_utils.py            # 配置、日志、checkpoint 保存
egno_eval_visualize.py       # 统一评估与可视化
eval_visualize_*.py          # 各数据集独立评估入口
```

## 3. EGNN 空间等变模块

### 3.1 `EGNN_Layer`

位置：`model/basic.py`

核心类：

```python
class EGNN_Layer(nn.Module):
    def forward(self, x, h, edge_index, edge_fea, v=None):
```

输入：

| 变量 | 形状 | 含义 |
|---|---:|---|
| `x` | `[N, 3]` | 节点坐标 |
| `h` | `[N, hidden]` | 节点标量特征 |
| `edge_index` | `[2, E]` | 有向边 |
| `edge_fea` | `[E, edge_dim]` | 边特征 |
| `v` | `[N, 3]` | 速度向量 |

代码中的消息构造：

```python
rij = x[row] - x[col]
hij = torch.cat((h[row], h[col], edge_fea), dim=-1)
message = self.edge_message_net(vectors=[rij], scalars=hij)
coord_message = self.coord_net(message)
f = (x[row] - x[col]) * coord_message
```

这里的关键是坐标更新只通过相对位移 `x_i - x_j` 与标量权重相乘，因此在旋转和平移变换下保持等变性。

节点坐标更新：

```python
tot_f = aggregate(f, row_index=row, n_node=x.shape[0], aggr='mean')
x = x + self.node_v_net(h) * v + tot_f
```

节点特征更新：

```python
tot_message = aggregate(message, row_index=row, n_node=x.shape[0], aggr='sum')
h = self.node_net(torch.cat((h, tot_message), dim=-1))
```

### 3.2 `InvariantScalarNet`

位置：`model/basic.py`

`InvariantScalarNet` 将向量输入转化为旋转不变的标量特征。核心思想是计算向量 Gram 矩阵：

```python
scalar = Z^T Z
```

在代码中：

```python
scalar = torch.einsum('bij,bjk->bik', Z_T, Z)
```

由于内积在旋转下不变，因此该模块适合作为边消息的标量网络。

## 4. 时间神经算子模块

### 4.1 时间嵌入

位置：`model/layer_no.py`

函数：

```python
get_timestep_embedding(timesteps, embedding_dim, max_positions=10000)
```

它使用类似 Transformer sinusoidal positional embedding 的方式，将离散时间步 `0, 1, ..., P-1` 编码为高维时间特征。

在 `EGNO.forward()` 中：

```python
time_emb = get_timestep_embedding(torch.arange(T), embedding_dim=self.time_emb_dim)
h = torch.cat((h, time_emb), dim=-1)
```

因此每个未来时间点都有不同的时间条件输入。

### 4.2 `TimeConv`

位置：`model/layer_no.py`

`TimeConv` 作用于节点标量特征序列：

```python
h: [T, N, hidden]
```

它通过 Fourier/spectral convolution 在时间维度上混合不同预测时间点的信息，使模型能够显式利用轨迹内部的时间相关性。

在 `EGNO.forward()` 中：

```python
h = time_conv(h.view(T, num_nodes, hidden_nf))
```

### 4.3 `TimeConv_x`

位置：`model/layer_no.py`

`TimeConv_x` 作用于几何向量序列：

```python
X = torch.stack((x_translated, v), dim=-1)
X: [T, N, 3, 2]
```

其中：

- 通道 0：去中心化后的坐标 `x - loc_mean`
- 通道 1：速度 `v`

代码：

```python
x_translated = x - loc_mean
X = torch.stack((x_translated, v), dim=-1)
temp = time_conv_x(X.view(T, num_nodes, 3, 2))
x = temp[..., 0].view(T * num_nodes, 3) + loc_mean
v = temp[..., 1].view(T * num_nodes, 3)
```

去中心化再加回 `loc_mean` 的处理，是为了在时间卷积中保持平移等变性。

## 5. EGNO 前向传播流程

位置：`model/egno.py`

核心类：

```python
class EGNO(EGNN):
```

前向函数：

```python
def forward(self, x, h, edge_index, edge_fea, v=None, loc_mean=None):
```

### 5.1 输入复制到多个时间点

EGNO 一次预测 `P` 个未来时刻。代码首先将初始状态复制到每个预测时间点：

```python
h = h.unsqueeze(0).repeat(T, 1, 1)
x = x.repeat(T, 1)
v = v.repeat(T, 1)
edge_fea = edge_fea.repeat(T, 1)
```

边索引也按时间维度复制并偏移：

```python
edges_0 = edge_index[0].repeat(T) + cumsum_edges
edges_1 = edge_index[1].repeat(T) + cumsum_edges
```

这样模型内部实际处理的是一个由 `T` 个时间切片拼接而成的大图。

### 5.2 时间卷积 + 空间消息传递

每一层 EGNO 都执行：

```python
for i in range(self.n_layers):
    if self.use_time_conv:
        h = TimeConv(h)
        x, v = TimeConv_x(x, v)
    x, v, h = EGNN_Layer(x, h, edge_index, edge_fea, v=v)
```

也就是说，每层都先进行时间方向的信息混合，再进行图结构上的空间消息传递。

### 5.3 输出

当输入包含速度 `v` 时：

```python
return x, v, h
```

其中 `x` 的形状为：

```text
[P * B * N, 3]
```

训练和评估时会还原为：

```text
[P, B, N, 3]
```

## 6. 不同数据集的模型配置

| 数据集 | `in_node_nf` | `in_edge_nf` | 层数 | hidden | `P` | 说明 |
|---|---:|---:|---:|---:|---:|---|
| N-body | 1 | 2 | 4 | 64 | 5 | 速度模长；电荷乘积 + 距离 |
| Motion | 2 | 2 | 6 | 128 | 5 | 速度模长 + 坐标特征；hop + 距离 |
| MD17 | 2 | 5 | 5 | 64 | 8 | 速度模长 + 原子类型；分子边特征 + 距离 |
| Protein | 2 | 2 | 4 | 128 | 4 | 速度模长 + 电荷；接触距离 + 距离 |

这些配置分别在以下文件中构造模型：

- `main_simulation_simple_no.py`
- `main_mocap_no.py`
- `main_md17_no.py`
- `main_mdanalysis_no.py`

## 7. 损失函数与评估指标

训练代码统一使用：

```python
loss_mse = nn.MSELoss(reduction='none')
losses = loss_mse(loc_pred, loc_end).view(P, B * N, 3)
losses = torch.mean(losses, dim=(1, 2))
loss = torch.mean(losses)
```

因此训练目标是：

```text
A-MSE = 所有预测时间点的位置 MSE 平均
```

日志中记录：

```python
res['loss'] += losses[-1].item() * batch_size
```

即最后一个时间点 MSE，对应论文中的：

```text
F-MSE
```

复现评估脚本 `egno_eval_visualize.py` 同时输出：

- `F-MSE`
- `A-MSE`
- `F-MSE_x1e2`
- `A-MSE_x1e2`
- `F-MAE`
- `A-MAE`
- `mean_final_node_l2`

## 8. 训练记录与 checkpoint 优化

原始训练脚本的记录较简洁，部分任务没有完整 checkpoint。复现中新增 `training_utils.py`，统一保存：

```text
logs/<exp_name>/
├── config.json
├── loss.json
├── summary.json
└── saved_model.pth
```

`saved_model.pth` 包含：

```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),  # 如果存在
    "args": vars(args),
    "best": best_metrics
}
```

这样可以支持：

- 复现实验配置追踪
- 最优模型恢复
- 后处理评估
- 可视化绘制

## 9. 可视化与评估实现

核心脚本：

```text
egno_eval_visualize.py
```

数据集独立入口：

```text
eval_visualize_simulation.py
eval_visualize_mocap.py
eval_visualize_md17.py
eval_visualize_mdanalysis.py
```

可视化脚本会自动：

1. 读取 `logs/<exp_name>/config.json`
2. 根据配置构造对应 dataset 和 EGNO 模型
3. 加载 `saved_model.pth`
4. 在 test split 上计算论文指标
5. 绘制输入状态、真实轨迹、预测轨迹、最终误差和时间误差曲线
6. 对若干单样本单独计算指标并绘图

输出目录示例：

```text
outputs/egno_md17_eval/
├── metrics.json
├── dataset_overview.png
├── training_curves.png
├── md17_exp_prediction.png
└── md17_exp_samples/
    ├── sample_metrics.json
    ├── md17_exp_sample_000.png
    └── ...
```

## 10. 对模型的理解与可改进方向

EGNO 的核心优势在于：它没有把未来轨迹拆成逐步 rollout，而是在一次前向中同时预测多个未来时间点，并用时间神经算子显式建模轨迹内部相关性。这能减少误差累积，也更符合物理系统中连续轨迹作为函数的观点。

可能的扩展方向包括：

1. **更复杂前向问题**：将 benchmark 扩展到流体、弹性体、可变拓扑粒子系统等更高复杂度动力学。
2. **更多反问题任务**：在给定部分观测轨迹时反推初始条件、材料参数或未知相互作用。
3. **几何先验增强**：在潜在空间或时间卷积中加入守恒量、局部参考系、约束投影等物理先验。
4. **时间卷积效率优化**：针对长时间序列探索低秩、稀疏频域或线性复杂度的时间算子。
5. **不确定性建模**：对多解或混沌动力学场景输出置信区间，而不仅是确定性点预测。

这些方向可以作为本复现项目后续研究延展。
