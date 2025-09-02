import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from src.federated.fedavg_client import FedAvgClient

def to_numpy_sd(sd):
    out={}
    for k,v in sd.items():
        if hasattr(v,'detach'): v=v.detach().cpu().numpy()
        elif hasattr(v,'numpy'): v=v.cpu().numpy() if hasattr(v,'cpu') else v.numpy()
        out[k]=v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--ckpt", default="models/flw/flower_run/global_round_200.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # 载入模型并评测
    sd = torch.load(Path(a.ckpt), map_location="cpu")["model_state_dict"]
    cli = FedAvgClient(port=a.port)
    cli.set_parameters(to_numpy_sd(sd))
    m = cli.evaluate()  # 可能返回 accuracy / reward 等占位/真实字段

    # 统一 success_rate
    sr = m.get("success_rate")
    if sr is None:
        sr = m.get("accuracy")
        if sr is None and m.get("total"):
            tot = float(m.get("total", 0))
            suc = float(m.get("success", 0))
            sr = (suc / tot) if tot > 0 else None

    # 统一 avg_reward
    rew = m.get("avg_reward")
    if rew is None:
        for k in ["reward", "reward_avg", "mean_reward", "avg_return", "return"]:
            if m.get(k) is not None:
                rew = m[k]; break
    try:
        if rew is not None and np.isnan(float(rew)): rew = None
    except Exception:
        pass

    out = {
        "port": a.port,
        "seed": a.seed,
        "samples": a.samples,
        "success_rate": None if sr is None else float(sr),
        "avg_reward": None if rew is None else float(rew),
        "ts": time.strftime("%F %T"),
        "returncode": 0,
        "note": "force_eval_to_json mapped fields (accuracy→success_rate etc.)"
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("WROTE", a.out)

if __name__ == "__main__":
    main()
