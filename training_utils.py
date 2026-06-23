import json
import os
from argparse import Namespace

import torch


def namespace_to_dict(args):
    if isinstance(args, Namespace):
        return vars(args).copy()
    return dict(args)


def ensure_exp_dir(args):
    exp_dir = os.path.join(args.outf, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=4, sort_keys=True)


def save_config(args, exp_dir):
    write_json(os.path.join(exp_dir, "config.json"), namespace_to_dict(args))


def init_history(args, task_name, metric_note):
    return {
        "task": task_name,
        "metric_note": metric_note,
        "config": namespace_to_dict(args),
        "train": [],
        "eval": [],
        "best": {},
    }


def save_history(history, exp_dir):
    write_json(os.path.join(exp_dir, "loss.json"), history)


def save_summary(history, exp_dir):
    write_json(os.path.join(exp_dir, "summary.json"), history.get("best", {}))


def save_checkpoint(path, epoch, model, optimizer, args, best, scheduler=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "args": namespace_to_dict(args),
        "best": best,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)
