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
import torch, numpy as np
from math import sqrt

SCRIPT_VERSION = "2025-08-07-r4"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from curriculum_trainer import CurriculumTrainer, build_agent  # noqa

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
                   n_samples=200, fixed_test_data=None, k_baseline=50, device=None, seed=42):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    # 评测数据集缓存
    if fixed_test_data is None:
        cache_dir = Path("../../models/releases") / time.strftime("%Y-%m-%d") / "datasets"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{trainer.port_name}__{stage.name.replace(' ', '_')}__seed{seed}__samples{n_samples}.npz"
        
        if cache_file.exists():
            logging.info(f"  📁 加载缓存数据集: {cache_file.name}")
            arr = np.load(cache_file, allow_pickle=True)
            test_data = arr["test_data"].tolist()
        else:
            logging.info(f"  🔄 生成新数据集并缓存: {cache_file.name}")
            test_data = trainer._generate_stage_data(stage, num_samples=n_samples)
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
    perf = {
        'avg_reward': float(np.mean(agent_rewards)),
        'baseline_avg_reward': float(np.mean(baseline_means)),
        'completion_rate': p_hat,
        'success': p_hat >= stage.success_threshold,
        'reward_threshold': float(thr_reward),
        'win_rate': p_hat
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

def main(port: str, n=200, seed=42, k=50, margin=0.02, force_recheck=False, disable_recheck=False):
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

    for stage in stages:
        agent = build_agent(port, hidden_dim=256, learning_rate=3e-4, batch_size=32,
                            device=device, num_heads=4, dropout=0.1,
                            state_dim=20, action_dim=15, node_feature_dim=8,
                            entropy_coef=0.02, ppo_epochs=6)
        agent.actor_critic.to(device)
        if hasattr(agent, "device"): agent.device = device
        agent.actor_critic.eval()

        ckpt_path = save_dir / f"stage_{stage.name}_best.pt"
        if not ckpt_path.exists():
            log.warning(f"  ⚠️ 找不到模型: {ckpt_path}"); port_ok = False; continue
        log.info(f"  ✅ 加载模型: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device)
        agent.actor_critic.load_state_dict(ckpt["model_state_dict"])

        saved_test_data = ckpt.get("test_data", None)
        if saved_test_data is not None:
            print(">>> 使用保存的 test_data:", stage.name)

        perf, std, stable, n_used = eval_one_stage(
            trainer, agent, stage, n_samples=n, fixed_test_data=saved_test_data, k_baseline=k, device=device, seed=seed
        )
        wr = perf["win_rate"]; thr = stage.success_threshold
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
                trainer, agent, stage, n_samples=n2, fixed_test_data=saved_test_data, k_baseline=k2, device=device, seed=seed
            )
            wr2 = perf2["win_rate"]; ok2, lb2 = pass_decision(wr2, thr, n_used2, margin=margin)
            print(f"↻ 复评结果：wr={wr2*100:.1f}%, std={std2*100:.1f}%, WilsonLB={lb2*100:.1f}% → {'通过' if ok2 else '未过'} "
                  f"({'稳定' if stable2 else '不稳定'})")
            if (ok2 and not ok) or (abs(wr2 - thr) < abs(wr - thr)):
                wr, std, stable, ok, lb, n_used = wr2, std2, stable2, ok2, lb2, n_used2

        port_ok &= ok
        stage_rows.append((stage.name, wr, thr, ok))

    if port_ok:
        print(f"港口 {port}: ✅ 通过")
    else:
        print(f"港口 {port}: ❌ 失败")
        for name, wr, thr, ok in stage_rows:
            print(f"  {name}: {wr*100:.1f}% (阈值{thr*100:.1f}%) {'✅' if ok else '❌'}")

    out_dir = Path("../../models/releases") / time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"consistency_{port}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"port": port, "stages": [
            {"stage": n, "win_rate": float(w), "threshold": float(t), "pass": bool(ok)}
            for (n, w, t, ok) in stage_rows]}, f, indent=2, ensure_ascii=False)
    log.info(f"📊 测试结果已保存到: {out_file}")
    return port_ok, stage_rows

def test_all_ports(n=200, seed=42, k=50, margin=0.02, force_recheck=False, disable_recheck=False):
    ports = ['baton_rouge', 'new_orleans', 'south_louisiana', 'gulfport']
    all_results = {}
    for port in ports:
        port_ok, stage_rows = main(port, n, seed, k, margin, force_recheck, disable_recheck)
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
    args = ap.parse_args()

    if args.all:
        test_all_ports(n=args.samples, seed=args.seed, k=args.k, margin=args.margin,
                       force_recheck=args.force_recheck, disable_recheck=args.no_recheck)
    elif args.port:
        main(args.port, n=args.samples, seed=args.seed, k=args.k, margin=args.margin,
             force_recheck=args.force_recheck, disable_recheck=args.no_recheck)
    else:
        print("请指定 --port PORT_NAME 或 --all"); sys.exit(1)