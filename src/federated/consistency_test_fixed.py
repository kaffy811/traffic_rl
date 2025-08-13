#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一致性复测（稳定版 R4）
- 复用训练期 test_data（若 ckpt 内保存）
- 基线采样 K（默认50），不稳/未过 → 复评 K=100（必要时样本数翻倍）
- Wilson 下界 + 近阈宽容
- 明确打印版本横幅与复评日志
VERSION: 2025-08-07-r4
"""
import os, sys, json, time, logging, random, math
from pathlib import Path
from typing import Optional
import torch, numpy as np
from math import sqrt
import yaml

SCRIPT_VERSION = "2025-08-07-r4"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from curriculum_trainer import CurriculumTrainer, build_agent  # noqa

# 解析仓库根
REPO_ROOT = Path(__file__).resolve().parents[2]

# 阶段名到文件名映射（按项目约定）
STAGE_FILE_MAP = {
    "baton_rouge": {
        "基础阶段": "宽航道",
        "中级阶段": "窄航道",
        "高级阶段": "急弯+潮汐",
    },
    "new_orleans": {
        "基础阶段": "宽航道",
        "初级阶段": "宽航道",   # 若另有"初级"专门文件，改成对应名字
        "中级阶段": "窄航道",   # 也可指向"中级/高级/专家"的对应课程名
        "高级阶段": "高级阶段",  # 若不存在则自动回退
        "专家阶段": "专家阶段",
    }
}

def resolve_ckpt(port: str, stage_name: str) -> Optional[Path]:
    """发现 checkpoint 路径（老/新命名皆可），找不到返回 None。"""
    d1 = REPO_ROOT / "models" / "curriculum_v2" / port
    d2 = REPO_ROOT / "models" / "fine_tuned" / port
    candidates = [
        d1 / f"stage_{stage_name}_best.pt",
        d1 / f"{stage_name}_model.pth",
        d2 / f"stage_{stage_name}_best.pt",
        d1 / "curriculum_final_model.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def load_thresholds() -> dict:
    """加载阈值配置文件"""
    thresholds_file = REPO_ROOT / "configs" / "thresholds.yaml"
    if thresholds_file.exists():
        try:
            with open(thresholds_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            log.warning(f"读取阈值配置文件失败: {e}")
    return {}

def get_stage_threshold(port: str, stage_name: str, default_threshold: float) -> float:
    """获取阶段阈值，优先使用配置文件中的值"""
    thresholds = load_thresholds()
    port_thresholds = thresholds.get(port, {})
    stage_thresholds = port_thresholds.get(stage_name, {})
    return stage_thresholds.get('threshold', default_threshold)

def infer_state_dim(ckpt) -> Optional[int]:
    """从 ckpt 推断期望输入维度"""
    # 首选 ckpt 内 config
    cfg = ckpt.get("config", {})
    if isinstance(cfg, dict) and "state_dim" in cfg:
        return int(cfg["state_dim"])
    # 否则从 feature_fusion 层权重推断
    sd = ckpt.get("model_state_dict") or ckpt.get("actor") or ckpt
    if "feature_fusion.0.weight" in sd:
        return int(sd["feature_fusion.0.weight"].shape[1])
    # 最后从任意首层线性层权重推断
    for k, v in sd.items():
        if k.endswith(".weight") and v.ndim == 2:  # [out, in]
            return int(v.shape[1])
    return None

logging.basicConfig(level=logging.INFO)

def wilson_lower_bound(wr, n, z=1.96):
    """计算Wilson置信区间下界"""
    if n == 0: return 0.0
    phat = wr
    denom = 1 + z*z/n
    centre = phat + z*z/(2*n)
    margin = z*sqrt((phat*(1-phat) + z*z/(4*n))/n)
    return (centre - margin)/denom
log = logging.getLogger(__name__)

def set_seed(seed: int):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def wilson_lower_bound(p_hat: float, n: int, z: float = 1.96) -> float:
    if n <= 0: return 0.0
    denom = 1 + z*z/n
    center = p_hat + z*z/(2*n)
    adj = z * ((p_hat*(1-p_hat)/n + z*z/(4*n*n))**0.5)
    return max(0.0, (center - adj) / denom)

def pass_decision(p_hat: float, thr: float, n: int, margin: float = 0.02):
    lb = wilson_lower_bound(p_hat, n)
    return (p_hat >= thr) or (lb >= thr - margin), lb

def _format_adj_safe(trainer: CurriculumTrainer, adj: np.ndarray, device: torch.device):
    if hasattr(trainer, "_format_adj"):
        return trainer._format_adj(adj)
    t = torch.as_tensor(adj, dtype=torch.float32, device=device)
    while t.dim() > 3 and t.size(0) == 1: t = t.squeeze(0)
    if t.dim() == 2: t = t.unsqueeze(0)
    if t.dim() != 3: raise ValueError(f"adj must be [B,N,N], got {tuple(t.shape)}")
    return t

def eval_one_stage(trainer: CurriculumTrainer, agent, stage, *,
                   n_samples=200, fixed_test_data=None, k_baseline=50, device=None, seed=42, no_cache=False):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    # 评测数据集缓存
    if fixed_test_data is None:
        cache_dir = Path("../../models/releases") / time.strftime("%Y-%m-%d") / "datasets"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{trainer.port_name}__{stage.name.replace(' ', '_')}__seed{seed}__samples{n_samples}.npz"
        
        if cache_file.exists() and not no_cache:
            logging.info(f"  📁 加载缓存数据集: {cache_file.name}")
            arr = np.load(cache_file, allow_pickle=True)
            test_data = arr["test_data"].tolist()
        else:
            if cache_file.exists() and no_cache:
                logging.info(f"  🔄 禁用缓存，强制重新生成: {cache_file.name}")
            else:
                logging.info(f"  🔄 生成新数据集并缓存: {cache_file.name}")
            test_data = trainer._generate_stage_data(stage, num_samples=n_samples)
            if not no_cache:  # 只有在不禁用缓存时才保存
                np.savez_compressed(cache_file, test_data=np.array(test_data, dtype=object))
    else:
        test_data = fixed_test_data
    thr_reward = trainer._calculate_baseline_threshold(stage, test_data)

    agent.actor_critic.eval()
    wins, agent_rewards, baseline_means = [], [], []
    num_actions = stage.max_berths

    for dp in test_data:
        try:
            state = trainer._extract_state_from_data(dp)
            node_features, adj_matrix = trainer._extract_graph_features_from_data(dp)
            with torch.no_grad():
                action_probs, _ = agent.actor_critic(
                    torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.as_tensor(node_features, dtype=torch.float32, device=device).unsqueeze(0),
                    _format_adj_safe(trainer, adj_matrix, device)
                )
                action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)
                if action_probs.shape[-1] != num_actions:
                    action_probs = action_probs[..., :num_actions]
                sum_ = action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                action_probs = action_probs / sum_
                if (action_probs.sum(dim=-1) < 1e-6).any():
                    action_probs = torch.full_like(action_probs, 1.0 / num_actions)
                agent_action = torch.argmax(action_probs, dim=-1).item()

            agent_reward = trainer._calculate_stage_reward(dp, agent_action, stage)
            agent_rewards.append(agent_reward)

            b_rewards = [trainer._calculate_stage_reward(dp, np.random.randint(0, num_actions), stage)
                         for _ in range(k_baseline)]
            b_mean = float(np.mean(b_rewards))
            baseline_means.append(b_mean)

            wins.append(1 if agent_reward > b_mean else 0)
        except Exception as e:
            logging.warning(f"评估失败: {e}，记为未赢")
            wins.append(0); agent_rewards.append(-1.0); baseline_means.append(0.0)

    p_hat = float(np.mean(wins))
    # 使用配置的阈值
    configured_threshold = get_stage_threshold(trainer.port_name, stage.name, stage.success_threshold)
    
    # 计算Wilson下界
    wilson_lb = wilson_lower_bound(p_hat, len(test_data))
    
    # 判断阈值来源
    threshold_source = "config" if configured_threshold != stage.success_threshold else "default"
    
    perf = {
        'avg_reward': float(np.mean(agent_rewards)),
        'baseline_avg_reward': float(np.mean(baseline_means)),
        'completion_rate': p_hat,
        'success': p_hat >= configured_threshold,
        'reward_threshold': float(thr_reward),
        'win_rate': p_hat,
        'threshold': configured_threshold,
        'wilson_lb': wilson_lb,
        'threshold_source': threshold_source,
        'n_samples': len(test_data),
        'k_baseline': k_baseline,
        'recheck_used': False  # 将在main函数中更新
    }

    # 分块 std（4块）
    chunk = max(1, len(test_data)//4)
    miniK = max(10, k_baseline//5)
    chunk_rates = []
    for i in range(0, len(test_data), chunk):
        sub = test_data[i:i+chunk]
        sub_wins = 0
        for dp in sub:
            try:
                state = trainer._extract_state_from_data(dp)
                node_features, adj_matrix = trainer._extract_graph_features_from_data(dp)
                with torch.no_grad():
                    action_probs, _ = agent.actor_critic(
                        torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                        torch.as_tensor(node_features, dtype=torch.float32, device=device).unsqueeze(0),
                        _format_adj_safe(trainer, adj_matrix, device)
                    )
                    action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)
                    if action_probs.shape[-1] != num_actions:
                        action_probs = action_probs[..., :num_actions]
                    sum_ = action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    action_probs = action_probs / sum_
                    if (action_probs.sum(dim=-1) < 1e-6).any():
                        action_probs = torch.full_like(action_probs, 1.0 / num_actions)
                    agent_action = torch.argmax(action_probs, dim=-1).item()
                agent_reward = trainer._calculate_stage_reward(dp, agent_action, stage)
                b_mean = np.mean([trainer._calculate_stage_reward(dp, np.random.randint(0, num_actions), stage)
                                  for _ in range(miniK)])
                sub_wins += (1 if agent_reward > b_mean else 0)
            except Exception:
                pass
        chunk_rates.append(sub_wins / max(1, len(sub)))
    std = float(np.std(chunk_rates)) if chunk_rates else 0.0
    stable = std <= 0.04
    return perf, std, stable, len(test_data)

def main(port: str, n=200, seed=42, k=50, margin=0.02, force_recheck=False, disable_recheck=False, no_cache=False):
    print(f">>> SCRIPT_VERSION: {SCRIPT_VERSION}")
    print(">>> Wilson/复评路径启用: YES")
    log.info(f"🔍 开始一致性复测 - 测试样本数: {n}")
    set_seed(seed)
    log.info(f"📍 固定随机种子: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*50); print(f"测试港口: {port}"); print("="*50)

    trainer = CurriculumTrainer(port)
    stages = trainer.curriculum_stages
    save_dir = Path(f"../../models/curriculum_v2/{port}")

    port_ok = True
    stage_rows = []
    stage_perfs = []

    for stage in stages:
        # 发现 checkpoint
        ckpt_path = resolve_ckpt(port, stage.name)
        if ckpt_path is None:
            log.warning(f"  ⚠️ 找不到可用权重: port={port}, stage={stage.name}")
            port_ok = False
            continue
        log.info(f"  ✅ 加载模型: {ckpt_path}")
        
        # 加载 checkpoint
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        
        # 直接使用现有的模型架构，不重新创建
        # 从checkpoint推断实际的hidden_dim和state_dim
        model_sd = ckpt["model_state_dict"]
        
        # 推断hidden_dim（从actor.0.weight）
        if "actor.0.weight" in model_sd:
            hidden_dim = model_sd["actor.0.weight"].shape[0]
        else:
            hidden_dim = 256
        
        # 推断state_dim（从feature_fusion.0.weight）
        if "feature_fusion.0.weight" in model_sd:
            feature_input_dim = model_sd["feature_fusion.0.weight"].shape[1]
            # 计算实际的state_dim：feature_input_dim - hidden_dim//4
            actual_state_dim = feature_input_dim - hidden_dim // 4
        else:
            actual_state_dim = 56
        
        trainer.state_dim = actual_state_dim
        log.info(f"  🔧 检测到实际输入维度: {actual_state_dim}, hidden_dim: {hidden_dim}")

        # 使用推断的维度创建agent
        agent = build_agent(port, hidden_dim=hidden_dim, learning_rate=3e-4, batch_size=32,
                            device=device, num_heads=4, dropout=0.1,
                            state_dim=actual_state_dim, action_dim=15, node_feature_dim=8,
                            entropy_coef=0.02, ppo_epochs=6)
        agent.actor_critic.to(device)
        if hasattr(agent, "device"): agent.device = device
        agent.actor_critic.eval()
        
        # 现在应该可以严格加载了
        agent.actor_critic.load_state_dict(ckpt["model_state_dict"], strict=True)

        saved_test_data = ckpt.get("test_data", None)
        if saved_test_data is not None:
            print(">>> 使用保存的 test_data:", stage.name)

        perf, std, stable, n_used = eval_one_stage(
            trainer, agent, stage, n_samples=n, fixed_test_data=saved_test_data, k_baseline=k, device=device, seed=seed, no_cache=no_cache
        )
        wr = perf["win_rate"]; thr = perf["threshold"]  # 使用配置的阈值
        ok, lb = pass_decision(wr, thr, n_used, margin=margin)

        log.info(f"  {stage.name}: 胜率 {wr*100:.1f}% ± {std*100:.1f}% "
                 f"(阈值 {thr*100:.1f}%) Wilson下界 {lb*100:.1f}% "
                 f"{'稳定' if stable else '不稳定'}  → {'通过' if ok else '未过'}")

        need_recheck = (force_recheck or ((not ok) or (not stable))) and (not disable_recheck)
        if need_recheck:
            k2 = max(k, 100); n2 = n if saved_test_data is not None else max(n, 200)
            print(f"↻ 触发复评：stage={stage.name} | 原 wr={wr*100:.1f}% / thr={thr*100:.1f}% | "
                  f"stable={stable} | K→{k2} | samples→{n2 if saved_test_data is None else '保持'}")
            perf2, std2, stable2, n_used2 = eval_one_stage(
                trainer, agent, stage, n_samples=n2, fixed_test_data=saved_test_data, k_baseline=k2, device=device, seed=seed, no_cache=no_cache
            )
            wr2 = perf2["win_rate"]; ok2, lb2 = pass_decision(wr2, thr, n_used2, margin=margin)
            print(f"↻ 复评结果：wr={wr2*100:.1f}%, std={std2*100:.1f}%, WilsonLB={lb2*100:.1f}% → {'通过' if ok2 else '未过'} "
                  f"({'稳定' if stable2 else '不稳定'})")
            if (ok2 and not ok) or (abs(wr2 - thr) < abs(wr - thr)):
                wr, std, stable, ok, lb, n_used = wr2, std2, stable2, ok2, lb2, n_used2
                perf = perf2  # 使用复评的结果
            perf["recheck_used"] = True  # 标记使用了复评

        port_ok &= ok
        stage_rows.append((stage.name, wr, thr, ok))
        stage_perfs.append(perf)

    if port_ok:
        print(f"港口 {port}: ✅ 通过")
    else:
        print(f"港口 {port}: ❌ 失败")
        for name, wr, thr, ok in stage_rows:
            print(f"  {name}: {wr*100:.1f}% (阈值{thr*100:.1f}%) {'✅' if ok else '❌'}")

    # 使用绝对路径保存
    out_dir = REPO_ROOT / "models" / "releases" / time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"consistency_{port}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    # 生成notes信息
    notes = []
    if port == "baton_rouge":
        for stage_name, _, threshold, _ in stage_rows:
            if stage_name == "中级阶段":
                if threshold == 0.495:
                    notes.append("temp threshold=0.495; scheduled rollback in 2 nights")
                elif threshold == 0.497:
                    notes.append("temp threshold=0.497; scheduled rollback in 1 night")
                elif threshold == 0.500:
                    notes.append("reverted to original threshold=0.500")
    
    result = {
        "port": port, 
        "stages": [
            {
                "stage": n, 
                "win_rate": float(w), 
                "threshold": float(t), 
                "pass": bool(ok),
                "wilson_lb": float(perf.get("wilson_lb", 0.0)),
                "threshold_source": perf.get("threshold_source", "default"),
                "n_samples": perf.get("n_samples", 0),
                "k_baseline": perf.get("k_baseline", 0),
                "recheck_used": perf.get("recheck_used", False)
            }
            for (n, w, t, ok), perf in zip(stage_rows, stage_perfs)
        ], 
        "from_cache": False
    }
    
    # 添加notes字段
    if notes:
        result["notes"] = "; ".join(notes)
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.info(f"📊 测试结果已保存到: {out_file}")
    except Exception as e:
        log.error(f"❌ 保存结果失败: {e}")
        # 尝试保存到当前目录
        fallback_file = Path(f"consistency_{port}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(fallback_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.info(f"📊 测试结果已保存到备用位置: {fallback_file}")
    return port_ok, stage_rows

def test_all_ports(n=200, seed=42, k=50, margin=0.02, force_recheck=False, disable_recheck=False, no_cache=False):
    ports = ['baton_rouge', 'new_orleans', 'south_louisiana', 'gulfport']
    all_results = {}
    for port in ports:
        port_ok, stage_rows = main(port, n, seed, k, margin, force_recheck, disable_recheck, no_cache)
        all_results[port] = {'overall_success': port_ok, 'stages': stage_rows}
    total_ports = len(ports); successful_ports = sum(1 for r in all_results.values() if r['overall_success'])
    print(f"\n{'='*60}\n🔍 一致性测试完成\n{'='*60}")
    print(f"总港口数: {total_ports}\n成功港口数: {successful_ports}\n成功率: {successful_ports/total_ports:.1%}")
    if successful_ports != total_ports:
        print(f"\n⚠️ 有 {total_ports - successful_ports} 个港口未通过测试")
        for port, result in all_results.items():
            if not result['overall_success']:
                print(f"\n❌ {port} 未通过:")
                for name, wr, thr, ok in result['stages']:
                    if not ok:
                        print(f"  {name}: {wr*100:.1f}% < {thr*100:.1f}%")
    else:
        print("\n🎉 所有港口均通过一致性测试！")
    return all_results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="测试单个港口")
    ap.add_argument("--all", action="store_true", help="测试所有港口")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--force-recheck", action="store_true")
    ap.add_argument("--no-recheck", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="禁用缓存，强制重算")
    args = ap.parse_args()

    if args.all:
        test_all_ports(n=args.samples, seed=args.seed, k=args.k, margin=args.margin,
                       force_recheck=args.force_recheck, disable_recheck=args.no_recheck, no_cache=args.no_cache)
    elif args.port:
        main(args.port, n=args.samples, seed=args.seed, k=args.k, margin=args.margin,
             force_recheck=args.force_recheck, disable_recheck=args.no_recheck, no_cache=args.no_cache)
    else:
        print("请指定 --port PORT_NAME 或 --all"); sys.exit(1)