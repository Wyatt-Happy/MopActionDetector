# utils.py
import os
import numpy as np
from typing import Union, List


def validate_video_path(video_path: Union[str, List[str]]) -> bool:
    """校验视频路径是否有效（支持单个/多个路径）"""
    if isinstance(video_path, str):
        video_path = [video_path]

    for path in video_path:
        if not os.path.exists(path):
            print(f"❌ 视频文件不存在：{path}")
            return False
        if not path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"⚠️ 非标准视频格式：{path}（建议mp4/avi/mov/mkv）")
    return True


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """标准化特征向量（去NaN/inf + 归一化）"""
    # 处理异常值
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
    # 归一化（保证余弦相似度可用点积计算）
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def generate_unique_id(prefix: str, idx: int) -> str:
    """生成唯一ID（前缀+时间戳+序号）"""
    import time
    return f"{prefix}_{int(time.time())}_{idx}"