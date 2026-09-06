# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import DetectionPredictor  # 预测
from .train import DetectionTrainer      # 训练
from .val import DetectionValidator      # 评估

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator"
