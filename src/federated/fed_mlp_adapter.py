# src/federated/fed_mlp_adapter.py
from __future__ import annotations
import os, math, random
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 训练器用于抽取 state/计算 reward
from .curriculum_trainer import CurriculumTrainer

def set_seed(seed: int):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, h1: int, h2: int, out_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.output = nn.Linear(h2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.output(x)  # [B, out_dim]

def _infer_dims_from_sd(sd: Dict[str, torch.Tensor]) -> Tuple[int,int,int,int]:
    """
    从 state_dict 推断 in_dim/h1/h2/out_dim：
      layer1.weight: [h1, in_dim]
      layer2.weight: [h2, h1]
      output.weight: [out, h2]
    """
    def shape_of(k): 
        t = sd.get(k, None)
        return tuple(t.shape) if t is not None else None

    s1 = shape_of("layer1.weight")
    s2 = shape_of("layer2.weight")
    so = shape_of("output.weight")

    if s1 and s2 and so and len(s1)==2 and len(s2)==2 and len(so)==2:
        h1, in_dim = s1
        h2, _      = s2
        out, _     = so
        return int(in_dim), int(h1), int(h2), int(out)

    # 退化兜底（尽量和你之前的模型一致）
    return 64, 128, 64, 16

def _fit_input(state_vec: np.ndarray, in_dim: int) -> torch.Tensor:
    """
    将任意长度的一维 state 向量对齐到 in_dim：
      - 长度不足：右侧零填充
      - 长度超出：截断
    返回 shape [in_dim]
    """
    x = np.asarray(state_vec, dtype=np.float32).reshape(-1)
    if x.size == in_dim:
        pass
    elif x.size < in_dim:
        pad = np.zeros(in_dim - x.size, dtype=np.float32)
        x = np.concatenate([x, pad], axis=0)
    else:
        x = x[:in_dim]
    return torch.from_numpy(x)

def _slice_or_pad_logits(logits: torch.Tensor, num_actions: int) -> torch.Tensor:
    """
    将网络输出维度对齐到 num_actions：
      - 若更长：切片 [:, :num_actions]
      - 若更短：右侧零填充
    """
    out_dim = logits.shape[-1]
    if out_dim == num_actions:
        return logits
    if out_dim > num_actions:
        return logits[..., :num_actions]
    # pad
    pad = torch.zeros(logits.size(0), num_actions - out_dim, device=logits.device, dtype=logits.dtype)
    return torch.cat([logits, pad], dim=-1)

def evaluate_with_ckpt(
    port: str,
    ckpt_path: str,
    *,
    samples: int = 120,
    seed: int = 42,
    k_baseline: int = 50,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    用联邦 MLP ckpt 在指定港口做快速直评（无图结构，仅基于state）。
    返回: {port, success_rate, avg_reward, num_samples, source}
    """
    set_seed(seed)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # 加载 ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model_state_dict") or ckpt  # 允许直接给 state_dict
    if not isinstance(sd, dict):
        raise ValueError("Invalid checkpoint: no model_state_dict/dict")

    in_dim, h1, h2, out_dim = _infer_dims_from_sd(sd)
    model = SimpleMLP(in_dim, h1, h2, out_dim).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f"[fed_mlp_adapter] load_state: missing={len(missing)} unexpected={len(unexpected)} "
              f"in={in_dim}, h1={h1}, h2={h2}, out={out_dim}")

    model.eval()

    # 课程/数据
    tr = CurriculumTrainer(port)
    stages = tr.curriculum_stages  # 遍历该港口的所有阶段
    wins: List[int] = []
    rewards: List[float] = []

    with torch.no_grad():
        for stage in stages:
            # 生成评测样本
            test_data = tr._generate_stage_data(stage, num_samples=samples)
            num_actions = int(stage.max_berths)

            for dp in test_data:
                try:
                    state_vec = tr._extract_state_from_data(dp)
                    x = _fit_input(state_vec, in_dim).to(device).unsqueeze(0)  # [1, in_dim]
                    logits = model(x)  # [1, out_dim]
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                    logits = _slice_or_pad_logits(logits, num_actions)        # [1, num_actions]

                    # 归一化防 NaN
                    probs = torch.softmax(logits, dim=-1)
                    if torch.isnan(probs).any() or (probs.sum(dim=-1) <= 1e-8).any():
                        probs = torch.ones_like(probs) / probs.shape[-1]

                    action = int(torch.argmax(probs, dim=-1).item())

                    # 计算奖励
                    agent_reward = tr._calculate_stage_reward(dp, action, stage)

                    # baseline 随机策略
                    br = []
                    for _ in range(k_baseline):
                        a = np.random.randint(0, num_actions)
                        br.append(tr._calculate_stage_reward(dp, a, stage))
                    b_mean = float(np.mean(br))

                    rewards.append(float(agent_reward))
                    wins.append(1 if agent_reward > b_mean else 0)
                except Exception as e:
                    if verbose:
                        print(f"[fed_mlp_adapter] eval error: {e}")
                    rewards.append(-1.0)
                    wins.append(0)

    n = len(rewards)
    sr = float(np.mean(wins)) if n > 0 else None
    avg_r = float(np.mean(rewards)) if n > 0 else None

    return {
        "port": port,
        "success_rate": sr,
        "avg_reward": avg_r,
        "num_samples": n,
        "source": ckpt_path,
    }

# 兼容旧导入名
def eval_with_ckpt(*args, **kwargs):
    return evaluate_with_ckpt(*args, **kwargs)
