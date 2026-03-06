# __init__.py
"""拖地行为检测模块包"""
from .config import MoppingDetectionConfig
from .feature_extractor import VideoFeatureExtractor
from .db_manager import EmbeddingDBManager
from .action_detector import MoppingActionDetector

# 导出核心类（外部使用时可直接 from mopping_detection import xxx）
__all__ = [
    "MoppingDetectionConfig",
    "VideoFeatureExtractor",
    "EmbeddingDBManager",
    "MoppingActionDetector"
]