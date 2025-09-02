#!/usr/bin/env python3
import os, sys, traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

print("PYTHONPATH OK:", REPO_ROOT)

try:
    from src.federated.eval_bridge import eval_port_via_consistency, eval_port_with_fed_mlp
    print("Imported eval_bridge ✓")
except Exception as e:
    print("Import eval_bridge FAILED:", e)
    traceback.print_exc()

try:
    from src.federated.fedavg_client import FedAvgClient
    print("Imported FedAvgClient ✓")
except Exception as e:
    print("Import FedAvgClient FAILED:", e)
    traceback.print_exc()

port = os.getenv("PORT","gulfport")
ckpt = os.getenv("CKPT","")
if not ckpt:
    # 猜一个 ckpt
    for tag in [os.getenv("BAL",""), os.getenv("UNB",""), os.getenv("FAIR","")]:
        if not tag: continue
        cands = list((REPO_ROOT/"models"/"flw"/tag).glob("global_round_*.pt"))
        if cands:
            ckpt = str(sorted(cands)[-1]); break

print("PORT:", port)
print("CKPT:", ckpt, "exists?", Path(ckpt).exists())

print("\n--- Try: eval_port_via_consistency ---")
try:
    r = eval_port_via_consistency(port, samples=int(os.getenv("FLW_EVAL_SAMPLES",120)))
    print("CONSISTENCY:", r)
except Exception as e:
    print("CONSISTENCY FAILED:", e)
    traceback.print_exc()

print("\n--- Try: eval_port_with_fed_mlp ---")
try:
    r = eval_port_with_fed_mlp(port, ckpt, samples=int(os.getenv("FLW_EVAL_SAMPLES",120)), verbose=True)
    print("CKPT_EVAL:", r)
except Exception as e:
    print("CKPT_EVAL FAILED:", e)
    traceback.print_exc()

print("\n--- Try: FedAvgClient.evaluate() ---")
try:
    os.environ["FLW_FORCE_CKPT"] = ckpt
    os.environ["FLW_EVAL_MODE"] = "ckpt"
    os.environ["FLW_EVAL_VERBOSE"] = "1"
    c = FedAvgClient(port=port)
    r = c.evaluate()
    print("CLIENT.evaluate() ->", r)
except Exception as e:
    print("CLIENT.evaluate FAILED:", e)
    traceback.print_exc()

