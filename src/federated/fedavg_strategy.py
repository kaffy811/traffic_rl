#!/usr/bin/env python3
"""
FedAvg策略实现
继承自Flower的FedAvg策略
"""

import flwr as fl
from typing import Dict, List, Optional, Tuple
import numpy as np

class FedAvgStrategy(fl.server.strategy.FedAvg):
    """FedAvg策略实现"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> Optional[fl.common.Parameters]:
        """聚合训练结果"""
        if not results:
            return None
        
        # 调用父类的聚合方法
        return super().aggregate_fit(server_round, results, failures)
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
    ) -> Optional[float]:
        """聚合评估结果"""
        if not results:
            return None
        
        # 调用父类的聚合方法
        return super().aggregate_evaluate(server_round, results, failures) 