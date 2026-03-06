# config.py
import torch


class MoppingDetectionConfig:
    """拖地检测全局配置类（所有可配置参数集中在这里）"""
    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 视频预处理配置
    FRAME_SIZE = (224, 224)  # 统一帧尺寸
    SAMPLE_FRAMES = 32  # 采样帧数
    NORMALIZE_MEAN = (0.485, 0.456, 0.406)  # ImageNet均值
    NORMALIZE_STD = (0.229, 0.224, 0.225)  # ImageNet标准差

    # 数据库配置
    DB_PATH = "E:/PoseYOLO/video_embedding_db"
    COLLECTION_NAME = "action_video_embeddings"
    COLLECTION_METADATA = {"description": "拖地/非拖地行为视频特征向量库"}

    # 检测阈值配置
    MOPPING_THRESHOLD = 0.75  # 正向相似度阈值
    SIMILARITY_GAP = 0.1  # 正向-反向相似度差值阈值
    TOP_K = 1  # 只匹配最相似的TOP1样本

    # 动作类型常量（避免硬编码字符串）
    ACTION_MOPPING = "mopping"
    ACTION_NON_MOPPING = "non_mopping"