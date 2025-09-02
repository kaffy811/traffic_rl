#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, time, argparse
from pathlib import Path

# 只依赖 bridge（成功则是真实评测），失败就写占位 JSON，避免导入 flwr/fedavg_client
from src.federated.eval_bridge import eval_port_with_fed_mlp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt")  # 可传，也可用 env: FLW_FORCE_CKPT
    args = ap.parse_args()

    # 让下游保持可复现
    os.environ["FLW_EVAL_SEED"] = str(args.seed)
    os.environ["FLW_EVAL_SAMPLES"] = str(args.samples)

    ckpt = args.ckpt or os.getenv("FLW_FORCE_CKpt") or os.getenv("FLW_FORCE_CKPT")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = None
    err_msg = None

    if ckpt and Path(ckpt).exists():
        try:
            # 关键：这里用“位置参数”传 ckpt（不要写 ckpt_path=...）
            metrics = eval_port_with_fed_mlp(args.port, ckpt, samples=args.samples, seed=args.seed, verbose=False)
        except TypeError as e:
            # 万一你的 bridge 恰好是新版、只认 ckpt_path=...，就再尝试一次
            try:
                metrics = eval_port_with_fed_mlp(args.port, ckpt_path=ckpt, samples=args.samples, seed=args.seed, verbose=False)  # type: ignore
            except Exception as e2:
                err_msg = f"{type(e2).__name__}: {e2}"
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
    else:
        err_msg = f"CKPT not found: {ckpt}"

    if not isinstance(metrics, dict):
        metrics = {}

    # 统一字段
    sr = metrics.get("success_rate") or metrics.get("win_rate") or metrics.get("accuracy")
    avg = metrics.get("avg_reward")

    record = {
        "port": args.port,
        "seed": args.seed,
        "samples": args.samples,
        "success_rate": float(sr) if sr is not None else None,
        "avg_reward": float(avg) if (isinstance(avg, (int, float)) and avg == avg) else None,
        "source": metrics.get("source"),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "returncode": 0 if sr is not None else 1,
    }
    if err_msg:
        record["error"] = err_msg

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print("WROTE", str(out_path))

if __name__ == "__main__":
    main()
