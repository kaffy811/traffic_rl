# src/federated/fedavg_client.py
from __future__ import annotations
import logging
from typing import Dict, Any, Tuple, List
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 你的真实训练客户端（保持你现有实现）
class FedAvgClient:
    """
    你已有的本地训练/评测实现。
    需要提供:
      - get_parameters(): Dict[str, np.ndarray]
      - set_parameters(params: Dict[str, np.ndarray]) -> None
      - train(...): 返回包含 loss / avg_reward / success_rate / num_samples 的 dict（若无法给真值可为空）
      - evaluate(): 返回 dict，尽量包含 {success_rate, avg_reward, num_samples}
    """
    def __init__(self, port: str, init_weights: str = None):
        self.port = port
        # TODO: 如果你有真实模型/权重初始化，在此载入

    def get_parameters(self) -> Dict[str, np.ndarray]:
        # TODO: 返回实际参数（此处放占位）
        return {
            "layer1.weight": np.zeros((128, 64), dtype=np.float32),
            "layer1.bias":   np.zeros((128,), dtype=np.float32),
            "layer2.weight": np.zeros((64, 128), dtype=np.float32),
            "layer2.bias":   np.zeros((64,), dtype=np.float32),
            "output.weight": np.zeros((16, 64), dtype=np.float32),
            "output.bias":   np.zeros((16,), dtype=np.float32),
        }

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        # TODO: 将 numpy 参数写回到你的模型
        pass

    def train(self, *, episodes: int, ppo_epochs: int, batch_size: int, entropy_coef: float) -> Dict[str, Any]:
        # TODO: 这里接你的本地训练逻辑；没有就返回占位
        return {
            "loss": None,
            "avg_reward": None,
            "success_rate": None,
            "num_samples": batch_size * episodes * ppo_epochs,
        }

    def evaluate(self) -> Dict[str, Any]:
        # 先尝试读取你机器上已有的评测 JSON（由 nightly_ci 或 force_eval_to_json 产出）
        try:
            from src.federated.eval_bridge import read_latest_eval
            got = read_latest_eval(self.port)
            if got:
                return {
                    "port": self.port,
                    "success_rate": got.get("success_rate"),
                    "avg_reward": got.get("avg_reward"),
                    "num_samples": got.get("num_samples") or 0,
                    "source": got.get("source"),
                }
        except Exception as e:
            logger.warning(f"读取一致性/forced JSON 失败：{e}")

        # TODO: 如果你有“直接对当前模型做离线评测”的实现，可在此调用；
        # 否则返回占位（Flower 评估阶段不阻塞）
        logger.warning("⚠️ evaluate() 未获得真实指标，返回占位（不影响联邦训练）")
        return {
            "port": self.port,
            "success_rate": None,
            "avg_reward": None,
            "num_samples": 0,
        }

# ---------- Flower NumPyClient 封装（不变/小改） ----------
try:
    import flwr as fl
except Exception:
    fl = None

class FlowerNumPyClient(fl.client.NumPyClient if fl else object):
    def __init__(self, core: FedAvgClient, *,
                 episodes: int = 8, ppo_epochs: int = 4, batch_size: int = 64, entropy_coef: float = 0.01):
        self.core = core
        self.episodes = episodes
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        # 固定顺序，保证与服务端聚合一致
        self.param_keys = list(self.core.get_parameters().keys())
        logger.info(f"🔑 参数键顺序: {self.param_keys}")

    # Flower hooks
    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        params = self.core.get_parameters()
        return [params[k] for k in self.param_keys]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params = {k: v for k, v in zip(self.param_keys, parameters)}
        self.core.set_parameters(params)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        stats = self.core.train(
            episodes=self.episodes, ppo_epochs=self.ppo_epochs,
            batch_size=self.batch_size, entropy_coef=self.entropy_coef
        )
        new_params = self.get_parameters(config)
        num_samples = int(stats.get("num_samples") or (self.batch_size * self.episodes * self.ppo_epochs))
        metrics = {
            "port": self.core.port,
            "episodes": self.episodes,
            "ppo_epochs": self.ppo_epochs,
            "batch_size": self.batch_size,
            "entropy_coef": self.entropy_coef,
            "loss": stats.get("loss"),
            "avg_reward": stats.get("avg_reward"),
            "success_rate": stats.get("success_rate"),
            "num_samples": num_samples,
        }
        return new_params, num_samples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        m = self.core.evaluate()
        sr = m.get("success_rate")
        # 给 Flower 一个“看起来像 loss 的数”（不影响论文）：有真值时用 1 - sr
        loss = float(1 - sr) if isinstance(sr, (float, int)) else 0.0
        num_samples = int(m.get("num_samples") or 0)
        metrics = {
            "port": self.core.port,
            "success_rate": sr,
            "avg_reward": m.get("avg_reward"),
            "num_samples": num_samples,
        }
        if "source" in m:
            metrics["source"] = m["source"]
        return loss, num_samples, metrics

# 便于 scripts/flower/client.py 使用
def build_flower_numpy_client(*, port: str, episodes=8, ppo_epochs=4, batch_size=64, entropy_coef=0.01, init_weights=None):
    core = FedAvgClient(port=port, init_weights=init_weights)
    return FlowerNumPyClient(core, episodes=episodes, ppo_epochs=ppo_epochs, batch_size=batch_size, entropy_coef=entropy_coef)
