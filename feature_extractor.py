# feature_extractor.py
import cv2
import torch
import numpy as np
from typing import List, Optional
from torchvision.transforms import Compose, Lambda, Resize, Normalize
from torchvision.models import resnet18, ResNet18_Weights
from config import MoppingDetectionConfig
from utils import normalize_embedding


class VideoFeatureExtractor:
    """视频特征提取器：专注于将视频转换为512维标准化特征向量"""

    def __init__(self, config: MoppingDetectionConfig = None):
        # 若未传入配置，使用默认配置
        self.config = config or MoppingDetectionConfig()
        self.device = torch.device(self.config.DEVICE)
        self.frame_transform = self._build_transform()
        self.feature_model = self._build_feature_model()

    def _build_transform(self) -> Compose:
        """构建帧预处理流水线"""
        return Compose([
            Resize(self.config.FRAME_SIZE, antialias=True),
            Lambda(lambda x: x / 255.0),
            Normalize(self.config.NORMALIZE_MEAN, self.config.NORMALIZE_STD)
        ])

    def _build_feature_model(self) -> torch.nn.Module:
        """构建特征提取模型（ResNet18，移除分类层）"""
        model = resnet18(weights=ResNet18_Weights.DEFAULT).to(self.device)
        model.eval()  # 推理模式（禁用Dropout/BatchNorm训练行为）
        # 移除最后一层全连接层，保留512维特征输出
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        return feature_extractor

    def read_frames(self, video_path: str) -> Optional[List[np.ndarray]]:
        """读取视频所有帧并转换为RGB格式"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频：{video_path}")
            return None

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV默认BGR → 转换为RGB（匹配PyTorch模型输入）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        if len(frames) == 0:
            print("❌ 视频无有效帧")
            return None
        return frames

    def sample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """等间距采样指定数量的帧"""
        sample_indices = np.linspace(
            0, len(frames) - 1, self.config.SAMPLE_FRAMES, dtype=int
        )
        return [frames[i] for i in sample_indices]

    def extract_frame_embedding(self, frame: np.ndarray) -> np.ndarray:
        """提取单帧的特征向量"""
        # 转换为Tensor并调整维度（HWC → CHW）
        frame_tensor = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1)
        # 预处理 + 增加batch维度
        frame_tensor = self.frame_transform(frame_tensor).unsqueeze(0).to(self.device)

        # 推理（禁用梯度计算，提升速度）
        with torch.no_grad():
            embedding = self.feature_model(frame_tensor)

        # 转换为numpy并压缩维度
        return embedding.cpu().numpy().squeeze()

    def extract_video_embedding(self, video_path: str) -> Optional[List[float]]:
        """提取视频的最终特征向量（帧均值 + 标准化）"""
        # 1. 读取并采样帧
        frames = self.read_frames(video_path)
        if frames is None:
            return None
        sampled_frames = self.sample_frames(frames)

        # 2. 提取每帧特征
        frame_embeddings = []
        for frame in sampled_frames:
            frame_emb = self.extract_frame_embedding(frame)
            frame_embeddings.append(frame_emb)

        # 3. 帧特征取均值 → 视频特征
        video_emb = np.mean(frame_embeddings, axis=0)
        # 4. 标准化
        video_emb = normalize_embedding(video_emb)

        # 5. 校验维度（必须512维）
        if len(video_emb) != 512:
            print(f"❌ 向量维度异常（期望512，实际{len(video_emb)}）")
            return None

        print(f"✅ 视频向量提取完成：{video_path}（512维）")
        return [float(v) for v in video_emb]