# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .tasks import (
    BaseModel,               # 所有模型的基础类
    ClassificationModel,     # 分类
    DetectionModel,          # 检测
    SegmentationModel,       # 分割
    guess_model_scale,       # 猜测模型规模 (n/s/m/l/x)
    guess_model_task,        # 猜测模型任务类型 (detect/segment/classify)
    load_checkpoint,         # 加载模型检查点
    parse_model,             # 关键函数: 解析YAML配置并构建模型
    torch_safe_load,         # 安全的PyTorch模型加载
    yaml_model_load,         # 从YAML文件加载模型配置
)

__all__ = (
    "load_checkpoint",
    "parse_model",
    "yaml_model_load",
    "guess_model_task",
    "guess_model_scale",
    "torch_safe_load",
    "DetectionModel",
    "SegmentationModel",
    "ClassificationModel",
    "BaseModel",
)
