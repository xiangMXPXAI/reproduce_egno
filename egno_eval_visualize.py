import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from model.egno import EGNO


TASK_INFO = {
    "simulation": {
        "title": "N-body charged particles",
        "input": "x0, v0, charge interactions",
        "output": "future particle positions",
        "solves": "learns a Coulomb-like 3D particle dynamics operator",
        "color": "#48cae4",
    },
    "mocap": {
        "title": "CMU motion capture",
        "input": "initial joint positions, velocities, skeleton graph",
        "output": "future 3D joint trajectories",
        "solves": "predicts human articulated motion on a body graph",
        "color": "#f72585",
    },
    "md17": {
        "title": "MD17 molecular dynamics",
        "input": "initial heavy-atom coordinates, velocities, atom types",
        "output": "future molecular conformations",
        "solves": "predicts molecular trajectory snapshots",
        "color": "#80ed99",
    },
    "mdanalysis": {
        "title": "Protein MDAnalysis ADK",
        "input": "initial backbone atoms, velocities, contact graph",
        "output": "future protein backbone conformations",
        "solves": "predicts protein conformational dynamics",
        "color": "#ffd166",
    },
}


def parse_args(default_experiments=None, default_output_dir=None):
    default_experiments = default_experiments or ["all"]
    default_output_dir = default_output_dir or "outputs/egno_eval_visualization"
    parser = argparse.ArgumentParser(
        description="Evaluate trained EGNO checkpoints and create dataset/prediction visualizations."
    )
    parser.add_argument("--logs-dir", type=str, default="logs")
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=default_experiments,
        help="Experiment folder names under logs, or 'all'.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=0, help="0 means use each config batch_size.")
    parser.add_argument("--max-batches", type=int, default=0, help="0 means evaluate the full test split.")
    parser.add_argument("--num-visual-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--max-plot-nodes", type=int, default=96)
    parser.add_argument("--max-plot-edges", type=int, default=700)
    parser.add_argument("--skip-failed", action="store_true", default=True)
    parser.add_argument("--no-skip-failed", dest="skip_failed", action="store_false")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def pick_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_experiments(logs_dir, names):
    logs_dir = Path(logs_dir)
    if names == ["all"]:
        exps = sorted(p for p in logs_dir.iterdir() if p.is_dir())
    else:
        exps = [logs_dir / name for name in names]
    return [p for p in exps if (p / "config.json").exists() and (p / "saved_model.pth").exists()]


def infer_task(exp_dir, cfg):
    name = exp_dir.name.lower()
    data_dir = str(cfg.get("data_dir", "")).lower()
    if "simulation" in name or "simulation" in data_dir:
        return "simulation"
    if "mocap" in name or "motion" in data_dir or "case" in cfg:
        return "mocap"
    if "md17" in name or "mol" in cfg:
        return "md17"
    if "mdanalysis" in name or cfg.get("load_cached") is not None or cfg.get("backbone") is not None:
        return "mdanalysis"
    raise ValueError(f"Cannot infer task for {exp_dir}")


def state_dict_from_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt
    return ckpt, {"best": {}}


def build_model(task, cfg, device):
    activation = nn.SiLU()
    if task == "simulation":
        return EGNO(
            n_layers=cfg["n_layers"],
            in_node_nf=1,
            in_edge_nf=2,
            hidden_nf=cfg["nf"],
            device=device,
            with_v=True,
            flat=cfg.get("flat", False),
            activation=activation,
            norm=cfg.get("norm", False),
            use_time_conv=True,
            num_modes=cfg.get("num_modes", 2),
            num_timesteps=cfg["num_timesteps"],
            time_emb_dim=cfg.get("time_emb_dim", 32),
        )
    if task == "mocap":
        return EGNO(
            n_layers=cfg["n_layers"],
            in_node_nf=2,
            in_edge_nf=2,
            hidden_nf=cfg["nf"],
            device=device,
            with_v=True,
            flat=cfg.get("flat", False),
            activation=activation,
            use_time_conv=True,
            num_modes=cfg.get("num_modes", 2),
            num_timesteps=cfg["num_timesteps"],
            time_emb_dim=cfg.get("time_emb_dim", 32),
        )
    if task == "md17":
        return EGNO(
            n_layers=cfg["n_layers"],
            in_node_nf=2,
            in_edge_nf=5,
            hidden_nf=cfg["nf"],
            device=device,
            with_v=True,
            flat=False,
            activation=activation,
            use_time_conv=cfg.get("use_time_conv", False),
            num_modes=cfg.get("num_modes", 2),
            num_timesteps=cfg["num_timesteps"],
            time_emb_dim=cfg.get("time_emb_dim", 32),
        )
    if task == "mdanalysis":
        return EGNO(
            n_layers=cfg["n_layers"],
            in_node_nf=2,
            in_edge_nf=2,
            hidden_nf=cfg["nf"],
            device=device,
            with_v=True,
            flat=cfg.get("flat", False),
            activation=activation,
            use_time_conv=True,
            num_modes=cfg.get("num_modes", 2),
            num_timesteps=cfg["num_timesteps"],
            time_emb_dim=cfg.get("time_emb_dim", 32),
        )
    raise ValueError(task)


def build_loader(task, cfg, batch_size):
    if task == "simulation":
        from simulation.dataset_simple import NBodyDynamicsDataset as SimulationDataset

        dataset = SimulationDataset(
            partition="test",
            data_dir=cfg["data_dir"],
            num_timesteps=cfg["num_timesteps"],
        )
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False), dataset

    if task == "mocap":
        from motion.dataset import MotionDynamicsDataset as MotionDataset

        dataset = MotionDataset(
            partition="test",
            max_samples=600,
            data_dir=cfg["data_dir"],
            delta_frame=cfg["delta_frame"],
            case=cfg.get("case", "walk"),
            num_timesteps=cfg["num_timesteps"],
        )
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False), dataset

    if task == "md17":
        from md17.dataset import MD17DynamicsDataset as MD17Dataset

        dataset = MD17Dataset(
            partition="test",
            max_samples=2000,
            data_dir=cfg["data_dir"],
            molecule_type=cfg["mol"],
            delta_frame=cfg["delta_frame"],
            num_timesteps=cfg["num_timesteps"],
        )
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False), dataset

    if task == "mdanalysis":
        from mdanalysis.dataset import MDDynamicsDataset as MDAnalysisDataset
        from mdanalysis.dataset import collate_mdd

        dataset = MDAnalysisDataset(
            "adk",
            partition="test",
            tmp_dir=cfg["data_dir"],
            delta_frame=cfg["delta_frame"],
            load_cached=cfg.get("load_cached", True),
            backbone=cfg.get("backbone", True),
            test_rot=False,
            test_trans=False,
            num_timesteps=cfg["num_timesteps"],
        )
        return (
            torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                collate_fn=collate_mdd,
            ),
            dataset,
        )

    raise ValueError(task)


def to_device(data, device):
    return [d.to(device) if torch.is_tensor(d) else d for d in data]


@torch.no_grad()
def predict_simulation(model, batch, dataset, cfg, device):
    loc, vel, edge_attr, charges, loc_end = to_device(batch, device)
    batch_size, n_nodes, _ = loc.size()
    loc_mean = loc.mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, 3)
    loc_flat = loc.view(-1, 3)
    vel_flat = vel.view(-1, 3)
    edge_attr = edge_attr.view(-1, edge_attr.shape[-1])
    target = loc_end.view(batch_size * n_nodes, cfg["num_timesteps"], 3)
    target = target.transpose(0, 1).contiguous().view(-1, 3)
    edges = dataset.get_edges(batch_size, n_nodes)
    edges = [edges[0].to(device), edges[1].to(device)]

    nodes = torch.sqrt(torch.sum(vel_flat ** 2, dim=1)).unsqueeze(1).detach()
    rows, cols = edges
    loc_dist = torch.sum((loc_flat[rows] - loc_flat[cols]) ** 2, dim=1).unsqueeze(1)
    edge_attr = torch.cat([edge_attr, loc_dist], dim=1).detach()
    pred, _, _ = model(loc_flat, nodes, edges, edge_attr, v=vel_flat, loc_mean=loc_mean)

    return {
        "loc0": loc.detach(),
        "vel0": vel.detach(),
        "target": unflatten_time(target, cfg["num_timesteps"], batch_size, n_nodes),
        "pred": unflatten_time(pred, cfg["num_timesteps"], batch_size, n_nodes),
        "edges": dataset.get_edges(1, n_nodes),
        "node_values": charges.detach(),
    }


@torch.no_grad()
def predict_mocap(model, batch, dataset, cfg, device):
    data = to_device(batch, device)
    batch_size, n_nodes, _ = data[0].size()
    for i in [-1, -2]:
        d = data[i].view(batch_size * n_nodes, cfg["num_timesteps"], 3)
        data[i] = d.transpose(0, 1).contiguous().view(-1, 3)

    loc, vel, edges, edge_attr, local_edges, local_edge_fea, z, loc_end, vel_end = data
    loc_mean = loc.mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, 3)
    loc_flat = loc.view(-1, 3)
    vel_flat = vel.view(-1, 3)
    offset = (torch.arange(batch_size, device=device) * n_nodes).unsqueeze(-1).unsqueeze(-1)
    edges = torch.cat(list(edges + offset), dim=-1)
    edge_attr = torch.cat(list(edge_attr), dim=0)
    z = z.view(-1, z.size(2))

    nodes = torch.sqrt(torch.sum(vel_flat ** 2, dim=1)).unsqueeze(1).detach()
    nodes = torch.cat((nodes, z / z.max().clamp(min=1e-12)), dim=-1)
    rows, cols = edges
    loc_dist = torch.sum((loc_flat[rows] - loc_flat[cols]) ** 2, dim=1).unsqueeze(1)
    edge_attr = torch.cat([edge_attr, loc_dist], dim=1).detach()
    pred, _, _ = model(loc_flat, nodes, [rows, cols], edge_attr, v=vel_flat, loc_mean=loc_mean)

    return {
        "loc0": loc.detach(),
        "vel0": vel.detach(),
        "target": unflatten_time(loc_end, cfg["num_timesteps"], batch_size, n_nodes),
        "pred": unflatten_time(pred, cfg["num_timesteps"], batch_size, n_nodes),
        "edges": [dataset.edges[0], dataset.edges[1]],
        "node_values": z.view(batch_size, n_nodes, -1).detach(),
    }


@torch.no_grad()
def predict_md17(model, batch, dataset, cfg, device):
    data, _cfg = batch[:-1], batch[-1]
    data = to_device(data, device)
    batch_size, n_nodes, _ = data[0].size()
    data = [d.view(-1, d.size(-1)) for d in data]
    for i in [4, 5]:
        d = data[i].view(batch_size * n_nodes, cfg["num_timesteps"], 3)
        data[i] = d.transpose(0, 1).contiguous().view(-1, 3)

    loc, vel, edge_attr, charges, loc_end, vel_end, z = data
    edges = dataset.get_edges(batch_size, n_nodes)
    edges = [edges[0].to(device), edges[1].to(device)]
    nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
    nodes = torch.cat((nodes, z / z.max().clamp(min=1e-12)), dim=-1)
    rows, cols = edges
    loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, dim=1).unsqueeze(1)
    edge_attr = torch.cat([edge_attr, loc_dist], dim=1).detach()
    loc_mean = loc.view(batch_size, n_nodes, 3).mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, 3)
    pred, _, _ = model(loc.detach(), nodes, edges, edge_attr, vel, loc_mean=loc_mean)

    return {
        "loc0": loc.view(batch_size, n_nodes, 3).detach(),
        "vel0": vel.view(batch_size, n_nodes, 3).detach(),
        "target": unflatten_time(loc_end, cfg["num_timesteps"], batch_size, n_nodes),
        "pred": unflatten_time(pred, cfg["num_timesteps"], batch_size, n_nodes),
        "edges": dataset.get_edges(1, n_nodes),
        "node_values": z.view(batch_size, n_nodes, -1).detach(),
    }


@torch.no_grad()
def predict_mdanalysis(model, batch, dataset, cfg, device):
    loc, vel, edges, edge_attr, local_edges, local_edge_fea, z, loc_end, vel_end = to_device(batch, device)
    batch_size, n_nodes, _ = loc.size()
    loc_mean = loc.mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, 3)
    loc_flat = loc.view(-1, 3)
    nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
    nodes = torch.cat((nodes, z / z.max().clamp(min=1e-12)), dim=-1)
    rows, cols = edges
    loc_dist = torch.sum((loc_flat[rows] - loc_flat[cols]) ** 2, dim=1).unsqueeze(1)
    edge_attr = torch.cat([edge_attr, loc_dist], dim=1).detach()
    pred, _, _ = model(loc_flat, nodes, [rows, cols], edge_attr, v=vel, loc_mean=loc_mean)

    first_edges = edges[:, (edges[0] < n_nodes) & (edges[1] < n_nodes)].detach().cpu()
    return {
        "loc0": loc.detach(),
        "vel0": vel.view(batch_size, n_nodes, 3).detach(),
        "target": unflatten_time(loc_end, cfg["num_timesteps"], batch_size, n_nodes),
        "pred": unflatten_time(pred, cfg["num_timesteps"], batch_size, n_nodes),
        "edges": [first_edges[0], first_edges[1]],
        "node_values": z.view(batch_size, n_nodes, -1).detach(),
    }


def unflatten_time(x, t_steps, batch_size, n_nodes):
    return x.view(t_steps, batch_size * n_nodes, 3).view(t_steps, batch_size, n_nodes, 3)


def predict_batch(task, model, batch, dataset, cfg, device):
    if task == "simulation":
        return predict_simulation(model, batch, dataset, cfg, device)
    if task == "mocap":
        return predict_mocap(model, batch, dataset, cfg, device)
    if task == "md17":
        return predict_md17(model, batch, dataset, cfg, device)
    if task == "mdanalysis":
        return predict_mdanalysis(model, batch, dataset, cfg, device)
    raise ValueError(task)


def update_metrics(acc, pred, target):
    diff = pred - target
    t_steps = pred.shape[0]
    batch_size = pred.shape[1]
    n_nodes = pred.shape[2]
    per_timestep_sq = diff.pow(2).sum(dim=(1, 2, 3)).detach().cpu().numpy()
    per_timestep_abs = diff.abs().sum(dim=(1, 2, 3)).detach().cpu().numpy()
    final_error = torch.linalg.norm(diff[-1], dim=-1)
    if acc["per_timestep_mse"] is None:
        acc["per_timestep_mse"] = np.zeros(t_steps, dtype=np.float64)
        acc["per_timestep_mae"] = np.zeros(t_steps, dtype=np.float64)
    acc["per_timestep_mse"] += per_timestep_sq
    acc["per_timestep_mae"] += per_timestep_abs
    acc["per_timestep_coord_count"] += batch_size * n_nodes * 3
    acc["all_sq_sum"] += diff.pow(2).sum().item()
    acc["all_abs_sum"] += diff.abs().sum().item()
    acc["all_coord_count"] += diff.numel()
    acc["final_sq_sum"] += diff[-1].pow(2).sum().item()
    acc["final_abs_sum"] += diff[-1].abs().sum().item()
    acc["final_coord_count"] += diff[-1].numel()
    acc["final_l2_sum"] += final_error.sum().item()
    acc["final_l2_sq_sum"] += final_error.pow(2).sum().item()
    acc["samples"] += batch_size
    acc["nodes"] += batch_size * n_nodes


def finalize_metrics(acc):
    nodes = max(acc["nodes"], 1)
    per_count = max(acc["per_timestep_coord_count"], 1)
    per_mse = acc["per_timestep_mse"] / per_count
    per_mae = acc["per_timestep_mae"] / per_count
    f_mse = acc["final_sq_sum"] / max(acc["final_coord_count"], 1)
    a_mse = acc["all_sq_sum"] / max(acc["all_coord_count"], 1)
    f_mae = acc["final_abs_sum"] / max(acc["final_coord_count"], 1)
    a_mae = acc["all_abs_sum"] / max(acc["all_coord_count"], 1)
    return {
        "num_batches": acc["num_batches"],
        "num_samples": acc["samples"],
        "num_nodes_counted": acc["nodes"],
        "F-MSE": float(f_mse),
        "A-MSE": float(a_mse),
        "F-MSE_x1e2": float(f_mse * 100.0),
        "A-MSE_x1e2": float(a_mse * 100.0),
        "F-RMSE": float(math.sqrt(max(f_mse, 0.0))),
        "A-RMSE": float(math.sqrt(max(a_mse, 0.0))),
        "F-MAE": float(f_mae),
        "A-MAE": float(a_mae),
        "mse_final_timestep": float(f_mse),
        "mse_all_timesteps": float(a_mse),
        "rmse_final_timestep": float(math.sqrt(max(f_mse, 0.0))),
        "mae_final_timestep": float(f_mae),
        "mean_final_node_l2": float(acc["final_l2_sum"] / nodes),
        "rms_final_node_l2": float(math.sqrt(acc["final_l2_sq_sum"] / nodes)),
        "per_timestep_mse": [float(x) for x in per_mse],
        "per_timestep_mae": [float(x) for x in per_mae],
    }


def sample_metrics(pred, target):
    diff = pred - target
    final = diff[-1]
    final_l2 = torch.linalg.norm(final, dim=-1)
    per_timestep_mse = diff.pow(2).mean(dim=(1, 2)).detach().cpu().numpy()
    per_timestep_mae = diff.abs().mean(dim=(1, 2)).detach().cpu().numpy()
    f_mse = final.pow(2).mean().item()
    a_mse = diff.pow(2).mean().item()
    return {
        "F-MSE": f_mse,
        "A-MSE": a_mse,
        "F-MSE_x1e2": f_mse * 100.0,
        "A-MSE_x1e2": a_mse * 100.0,
        "F-MAE": final.abs().mean().item(),
        "A-MAE": diff.abs().mean().item(),
        "mean_final_node_l2": final_l2.mean().item(),
        "max_final_node_l2": final_l2.max().item(),
        "per_timestep_mse": [float(x) for x in per_timestep_mse],
        "per_timestep_mae": [float(x) for x in per_timestep_mae],
    }


def slice_visual_sample(visual, sample_idx):
    item = {}
    for key, value in visual.items():
        if key in {"edges"}:
            item[key] = value
        elif torch.is_tensor(value):
            if value.dim() >= 2 and key in {"loc0", "vel0", "node_values"}:
                item[key] = value[sample_idx:sample_idx + 1].detach().cpu()
            elif value.dim() >= 3 and key in {"target", "pred"}:
                item[key] = value[:, sample_idx:sample_idx + 1].detach().cpu()
            else:
                item[key] = value.detach().cpu()
        else:
            item[key] = value
    return item


def make_axes_equal(ax, xyz):
    xyz = np.asarray(xyz)
    if xyz.size == 0:
        return
    mins = xyz.reshape(-1, 3).min(axis=0)
    maxs = xyz.reshape(-1, 3).max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def style_3d(ax):
    ax.set_facecolor("#0b1020")
    ax.grid(False)
    ax.xaxis.pane.set_facecolor((0.05, 0.07, 0.12, 1.0))
    ax.yaxis.pane.set_facecolor((0.05, 0.07, 0.12, 1.0))
    ax.zaxis.pane.set_facecolor((0.05, 0.07, 0.12, 1.0))
    ax.tick_params(colors="#9aa4b2", labelsize=7)
    ax.set_xlabel("x", color="#9aa4b2", labelpad=-2)
    ax.set_ylabel("y", color="#9aa4b2", labelpad=-2)
    ax.set_zlabel("z", color="#9aa4b2", labelpad=-2)


def choose_nodes(n_nodes, max_nodes, values=None):
    if n_nodes <= max_nodes:
        return np.arange(n_nodes)
    if values is not None:
        values = np.asarray(values).reshape(-1)
        order = np.argsort(values)[-max_nodes:]
        return np.sort(order)
    return np.linspace(0, n_nodes - 1, max_nodes).astype(int)


def sampled_edges(edges, node_mask, max_edges):
    rows, cols = edges
    rows = rows.detach().cpu().numpy() if torch.is_tensor(rows) else np.asarray(rows)
    cols = cols.detach().cpu().numpy() if torch.is_tensor(cols) else np.asarray(cols)
    keep = np.isin(rows, node_mask) & np.isin(cols, node_mask)
    edge_arr = np.stack([rows[keep], cols[keep]], axis=1)
    if len(edge_arr) > max_edges:
        idx = np.linspace(0, len(edge_arr) - 1, max_edges).astype(int)
        edge_arr = edge_arr[idx]
    return edge_arr


def plot_edges(ax, coords, edge_arr, color="#3b82f6", alpha=0.18, lw=0.55):
    for i, j in edge_arr:
        p, q = coords[int(i)], coords[int(j)]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=color, alpha=alpha, lw=lw)


def plot_initial(ax, loc0, vel0, edges, nodes, max_edges):
    style_3d(ax)
    edge_arr = sampled_edges(edges, nodes, max_edges)
    plot_edges(ax, loc0, edge_arr)
    colors = np.linalg.norm(vel0[nodes], axis=1)
    ax.scatter(loc0[nodes, 0], loc0[nodes, 1], loc0[nodes, 2], c=colors, s=18, cmap="viridis", depthshade=True)
    if len(nodes) <= 80:
        v = vel0[nodes]
        scale = np.linalg.norm(loc0.max(axis=0) - loc0.min(axis=0)) * 0.045
        denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        v = v / denom * scale
        ax.quiver(loc0[nodes, 0], loc0[nodes, 1], loc0[nodes, 2], v[:, 0], v[:, 1], v[:, 2],
                  color="#f8fafc", alpha=0.55, linewidth=0.7)
    make_axes_equal(ax, loc0[nodes])
    ax.set_title("Input: initial state", color="#e5e7eb", pad=8, fontsize=11, weight="bold")


def plot_trajectories(ax, target, pred, nodes, task_color):
    style_3d(ax)
    t = target[:, nodes, :]
    p = pred[:, nodes, :]
    for k, node in enumerate(nodes):
        alpha = 0.9 if k < 32 else 0.35
        ax.plot(t[:, k, 0], t[:, k, 1], t[:, k, 2], color="#e5e7eb", lw=1.25, alpha=alpha)
        ax.plot(p[:, k, 0], p[:, k, 1], p[:, k, 2], color=task_color, lw=1.25, ls="--", alpha=alpha)
    ax.scatter(target[-1, nodes, 0], target[-1, nodes, 1], target[-1, nodes, 2],
               s=16, color="#e5e7eb", alpha=0.9, label="Ground truth")
    ax.scatter(pred[-1, nodes, 0], pred[-1, nodes, 1], pred[-1, nodes, 2],
               s=16, color=task_color, alpha=0.9, label="Prediction")
    make_axes_equal(ax, np.concatenate([t.reshape(-1, 3), p.reshape(-1, 3)], axis=0))
    ax.set_title("Output: future trajectory", color="#e5e7eb", pad=8, fontsize=11, weight="bold")


def plot_error(ax, target, pred, nodes):
    style_3d(ax)
    err = np.linalg.norm(pred[-1] - target[-1], axis=1)
    sc = ax.scatter(target[-1, nodes, 0], target[-1, nodes, 1], target[-1, nodes, 2],
                    c=err[nodes], s=22, cmap="magma", depthshade=True)
    make_axes_equal(ax, target[-1, nodes])
    ax.set_title("Final-step spatial error", color="#e5e7eb", pad=8, fontsize=11, weight="bold")
    return sc


def plot_text_card(ax, exp_name, task, cfg, metrics, sample_label=None):
    info = TASK_INFO[task]
    ax.set_facecolor("#0b1020")
    ax.axis("off")
    lines = [
        info["title"],
        "",
        f"Experiment: {exp_name}" if sample_label is None else f"Experiment: {exp_name} | {sample_label}",
        f"Input: {info['input']}",
        f"Output: {info['output']}",
        f"Task: {info['solves']}",
        "",
        f"T steps: {cfg.get('num_timesteps')}",
        f"F-MSE: {metrics['F-MSE']:.6g}",
        f"A-MSE: {metrics['A-MSE']:.6g}",
        f"F-MSE x1e2: {metrics['F-MSE_x1e2']:.6g}",
        f"F-MAE: {metrics['F-MAE']:.6g}",
        f"Mean node L2: {metrics['mean_final_node_l2']:.6g}",
    ]
    ax.text(0.04, 0.93, lines[0], color=info["color"], fontsize=16, weight="bold", va="top")
    ax.text(0.04, 0.82, "\n".join(lines[2:]), color="#d1d5db", fontsize=10.2, va="top", linespacing=1.65)


def plot_per_timestep(ax, metrics, task_color):
    ax.set_facecolor("#0b1020")
    x = np.arange(1, len(metrics["per_timestep_mse"]) + 1)
    ax.plot(x, metrics["per_timestep_mse"], marker="o", color=task_color, lw=2.0, label="MSE")
    ax.plot(x, metrics["per_timestep_mae"], marker="s", color="#e5e7eb", lw=1.5, label="MAE")
    ax.set_xlabel("Predicted timestep", color="#cbd5e1")
    ax.set_ylabel("Error", color="#cbd5e1")
    ax.tick_params(colors="#9aa4b2")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, color="#334155", alpha=0.35, lw=0.7)
    ax.legend(facecolor="#111827", edgecolor="#334155", labelcolor="#e5e7eb", fontsize=8)
    ax.set_title("Temporal error profile", color="#e5e7eb", fontsize=11, weight="bold")


def plot_prediction_figure(
    exp_dir, task, cfg, visual, metrics, output_dir, dpi, max_nodes, max_edges, sample_index=None
):
    info = TASK_INFO[task]
    exp_name = exp_dir.name

    loc0 = visual["loc0"][0].detach().cpu().numpy()
    vel0 = visual["vel0"][0].detach().cpu().numpy()
    target = visual["target"][:, 0].detach().cpu().numpy()
    pred = visual["pred"][:, 0].detach().cpu().numpy()
    final_err = np.linalg.norm(pred[-1] - target[-1], axis=1)
    nodes = choose_nodes(loc0.shape[0], max_nodes, values=final_err)

    fig = plt.figure(figsize=(17, 10), facecolor="#050816")
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1.05, 1.35, 1.35, 1.35], height_ratios=[1, 1])

    ax_card = fig.add_subplot(gs[:, 0])
    sample_label = None if sample_index is None else f"sample {sample_index}"
    plot_text_card(ax_card, exp_name, task, cfg, metrics, sample_label=sample_label)

    ax_initial = fig.add_subplot(gs[0, 1], projection="3d")
    plot_initial(ax_initial, loc0, vel0, visual["edges"], nodes, max_edges)

    ax_traj = fig.add_subplot(gs[0, 2], projection="3d")
    plot_trajectories(ax_traj, target, pred, nodes, info["color"])
    handles = [
        Line2D([0], [0], color="#e5e7eb", lw=2, label="Ground truth"),
        Line2D([0], [0], color=info["color"], lw=2, ls="--", label="Prediction"),
    ]
    ax_traj.legend(handles=handles, loc="upper left", fontsize=8, facecolor="#111827",
                   edgecolor="#334155", labelcolor="#e5e7eb")

    ax_error = fig.add_subplot(gs[0, 3], projection="3d")
    sc = plot_error(ax_error, target, pred, nodes)
    cbar = fig.colorbar(sc, ax=ax_error, shrink=0.58, pad=0.03)
    cbar.ax.tick_params(colors="#cbd5e1", labelsize=8)
    cbar.outline.set_edgecolor("#334155")

    ax_steps = fig.add_subplot(gs[1, 1:3])
    plot_per_timestep(ax_steps, metrics, info["color"])

    ax_final = fig.add_subplot(gs[1, 3], projection="3d")
    style_3d(ax_final)
    ax_final.scatter(target[-1, nodes, 0], target[-1, nodes, 1], target[-1, nodes, 2],
                     s=22, color="#e5e7eb", alpha=0.85)
    ax_final.scatter(pred[-1, nodes, 0], pred[-1, nodes, 1], pred[-1, nodes, 2],
                     s=22, color=info["color"], alpha=0.85)
    for node in nodes[:80]:
        a = target[-1, node]
        b = pred[-1, node]
        ax_final.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#fb7185", alpha=0.42, lw=0.75)
    make_axes_equal(ax_final, np.concatenate([target[-1, nodes], pred[-1, nodes]], axis=0))
    ax_final.set_title("Prediction displacement vectors", color="#e5e7eb", pad=8, fontsize=11, weight="bold")

    fig.suptitle(f"EGNO Evaluation | {info['title']} | {exp_name}",
                 color="#f8fafc", fontsize=18, weight="bold", y=0.985)
    fig.subplots_adjust(left=0.035, right=0.98, top=0.93, bottom=0.06, wspace=0.24, hspace=0.18)
    suffix = "prediction" if sample_index is None else f"sample_{sample_index:03d}"
    path = output_dir / f"{exp_name}_{suffix}.png"
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def read_loss_history(path):
    if not path.exists():
        return None
    data = load_json(path)
    if "train" in data and "eval" in data:
        train = data.get("train", [])
        evals = data.get("eval", [])
        return {
            "train_epoch": [x.get("epoch") for x in train],
            "train_loss": [x.get("loss") for x in train],
            "eval_epoch": [x.get("epoch") for x in evals],
            "val_loss": [x.get("val_loss") for x in evals],
            "test_loss": [x.get("test_loss") for x in evals],
        }
    return {
        "train_epoch": list(range(len(data.get("train loss", [])))),
        "train_loss": data.get("train loss", []),
        "eval_epoch": data.get("eval epoch", data.get("epochs", [])),
        "val_loss": data.get("val loss", []),
        "test_loss": data.get("test loss", data.get("loss", [])),
    }


def plot_loss_curves(experiments, output_dir, dpi):
    if not experiments:
        return None
    cols = 2
    rows = math.ceil(len(experiments) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.2 * rows), facecolor="#050816")
    axes = np.asarray(axes).reshape(-1)
    for ax, item in zip(axes, experiments):
        exp_dir, task = item["exp_dir"], item["task"]
        hist = read_loss_history(exp_dir / "loss.json")
        info = TASK_INFO[task]
        ax.set_facecolor("#0b1020")
        if hist is not None:
            if hist["train_loss"]:
                ax.plot(hist["train_epoch"], hist["train_loss"], color="#94a3b8", alpha=0.75, lw=1.2, label="train")
            if hist["val_loss"]:
                ax.plot(hist["eval_epoch"], hist["val_loss"], color=info["color"], lw=2.0, label="val")
            if hist["test_loss"]:
                ax.plot(hist["eval_epoch"], hist["test_loss"], color="#f8fafc", lw=1.4, ls="--", label="test")
        ax.set_title(exp_dir.name, color="#e5e7eb", fontsize=12, weight="bold")
        ax.set_xlabel("epoch", color="#cbd5e1")
        ax.set_ylabel("loss", color="#cbd5e1")
        ax.tick_params(colors="#9aa4b2")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, color="#334155", alpha=0.32)
        ax.legend(facecolor="#111827", edgecolor="#334155", labelcolor="#e5e7eb", fontsize=8)
    for ax in axes[len(experiments):]:
        ax.axis("off")
    fig.suptitle("EGNO Training Curves", color="#f8fafc", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "training_curves.png"
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_dataset_overview(results, output_dir, dpi):
    fig, ax = plt.subplots(figsize=(15, 7), facecolor="#050816")
    ax.set_facecolor("#050816")
    ax.axis("off")
    ax.text(0.03, 0.93, "EGNO Dataset and Task Overview", color="#f8fafc",
            fontsize=22, weight="bold", va="top")
    x0s = np.linspace(0.04, 0.76, max(len(results), 1))
    width = 0.20 if len(results) <= 4 else 0.16
    for idx, (name, item) in enumerate(results.items()):
        task = item["task"]
        info = TASK_INFO[task]
        x0 = x0s[idx]
        y0 = 0.15
        rect = plt.Rectangle((x0, y0), width, 0.62, facecolor="#0b1020",
                             edgecolor=info["color"], linewidth=1.6, alpha=0.96)
        ax.add_patch(rect)
        ax.text(x0 + 0.018, y0 + 0.55, info["title"], color=info["color"],
                fontsize=13, weight="bold")
        lines = [
            f"Run: {name}",
            f"Input: {info['input']}",
            f"Output: {info['output']}",
            f"T: {item['config'].get('num_timesteps')}",
            f"F-MSE: {item['metrics']['F-MSE']:.4g}",
            f"A-MSE: {item['metrics']['A-MSE']:.4g}",
            f"F-MSE x1e2: {item['metrics']['F-MSE_x1e2']:.4g}",
        ]
        ax.text(x0 + 0.018, y0 + 0.48, "\n".join(lines), color="#d1d5db",
                fontsize=9.3, va="top", linespacing=1.55)
    path = output_dir / "dataset_overview.png"
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def evaluate_experiment(exp_dir, args, device):
    cfg = load_json(exp_dir / "config.json")
    task = infer_task(exp_dir, cfg)
    batch_size = args.batch_size or int(cfg.get("batch_size", 1))
    loader, dataset = build_loader(task, cfg, batch_size)

    model = build_model(task, cfg, device)
    state, ckpt = state_dict_from_checkpoint(exp_dir / "saved_model.pth", device)
    model.load_state_dict(state, strict=True)
    model.eval()

    acc = {
        "num_batches": 0,
        "samples": 0,
        "nodes": 0,
        "per_timestep_mse": None,
        "per_timestep_mae": None,
        "per_timestep_coord_count": 0,
        "all_sq_sum": 0.0,
        "all_abs_sum": 0.0,
        "all_coord_count": 0,
        "final_sq_sum": 0.0,
        "final_abs_sum": 0.0,
        "final_coord_count": 0,
        "final_l2_sum": 0.0,
        "final_l2_sq_sum": 0.0,
    }
    visual = None
    sample_visuals = []
    sample_records = []
    global_sample_idx = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            out = predict_batch(task, model, batch, dataset, cfg, device)
            update_metrics(acc, out["pred"], out["target"])
            acc["num_batches"] += 1
            if visual is None:
                visual = out
            batch_samples = out["pred"].shape[1]
            for local_idx in range(batch_samples):
                if len(sample_visuals) < args.num_visual_samples:
                    item = slice_visual_sample(out, local_idx)
                    sm = sample_metrics(item["pred"][:, 0], item["target"][:, 0])
                    sm["sample_index"] = global_sample_idx
                    sample_visuals.append(item)
                    sample_records.append(sm)
                global_sample_idx += 1
            if args.max_batches and acc["num_batches"] >= args.max_batches:
                break

    metrics = finalize_metrics(acc)
    metrics["checkpoint_best"] = ckpt.get("best", {}) if isinstance(ckpt, dict) else {}
    metrics["task"] = task
    fig_path = plot_prediction_figure(
        exp_dir=exp_dir,
        task=task,
        cfg=cfg,
        visual=visual,
        metrics=metrics,
        output_dir=Path(args.output_dir),
        dpi=args.dpi,
        max_nodes=args.max_plot_nodes,
        max_edges=args.max_plot_edges,
    )
    sample_dir = Path(args.output_dir) / f"{exp_dir.name}_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for item, record in zip(sample_visuals, sample_records):
        plot_prediction_figure(
            exp_dir=exp_dir,
            task=task,
            cfg=cfg,
            visual=item,
            metrics=record,
            output_dir=sample_dir,
            dpi=args.dpi,
            max_nodes=args.max_plot_nodes,
            max_edges=args.max_plot_edges,
            sample_index=record["sample_index"],
        )
    write_json(sample_dir / "sample_metrics.json", sample_records)
    return {
        "task": task,
        "config": cfg,
        "metrics": metrics,
        "sample_metrics": sample_records,
        "figure": str(fig_path),
        "exp_dir": exp_dir,
    }


def main(default_experiments=None, default_output_dir=None):
    args = parse_args(default_experiments=default_experiments, default_output_dir=default_output_dir)
    set_seed(args.seed)
    device = pick_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_dirs = discover_experiments(args.logs_dir, args.experiments)
    if not exp_dirs:
        raise RuntimeError(f"No experiments found in {args.logs_dir}")

    results = {}
    finished = []
    for exp_dir in exp_dirs:
        try:
            print(f"[EGNO] evaluating {exp_dir.name} on {device} ...")
            item = evaluate_experiment(exp_dir, args, device)
            results[exp_dir.name] = {
                "task": item["task"],
                "config": item["config"],
                "metrics": item["metrics"],
                "sample_metrics": item["sample_metrics"],
                "figure": item["figure"],
            }
            finished.append({"exp_dir": exp_dir, "task": item["task"]})
            print(
                f"[EGNO] done {exp_dir.name}: "
                f"F-MSE={item['metrics']['F-MSE']:.6g}, A-MSE={item['metrics']['A-MSE']:.6g}"
            )
        except Exception as exc:
            if not args.skip_failed:
                raise
            results[exp_dir.name] = {"error": repr(exc)}
            print(f"[EGNO] skipped {exp_dir.name}: {exc}")

    plot_loss_curves(finished, output_dir, args.dpi)
    plot_dataset_overview({k: v for k, v in results.items() if "metrics" in v}, output_dir, args.dpi)
    write_json(output_dir / "metrics.json", results)
    print(f"[EGNO] outputs saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
