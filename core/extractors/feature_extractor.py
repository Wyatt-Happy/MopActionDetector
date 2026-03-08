# feature_extractor.py
import cv2
import torch
import torch.quantization
import numpy as np
from typing import List, Optional, Union
from torchvision.transforms import Compose, Lambda, Resize, Normalize
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from core.utils.config import MoppingDetectionConfig
from core.utils.utils import normalize_embedding
from core.managers.logger import get_logger

logger = get_logger(__name__)


class VideoFeatureExtractor:
    """视频特征提取器：支持 ResNet18/50，可选 INT8 量化"""

    def __init__(self, config: MoppingDetectionConfig = None):
        self.config = config or MoppingDetectionConfig()
        self.device = torch.device(self.config.DEVICE)
        self.frame_transform = self._build_transform()
        self.feature_model = self._build_feature_model()
        logger.info(f"特征提取器初始化完成，使用模型: {self.config.MODEL_BACKBONE}")

    def _build_transform(self) -> Compose:
        """构建帧预处理流水线"""
        return Compose([
            Resize(self.config.FRAME_SIZE, antialias=True),
            Lambda(lambda x: x / 255.0),
            Normalize(self.config.NORMALIZE_MEAN, self.config.NORMALIZE_STD)
        ])

    def _build_feature_model(self) -> torch.nn.Module:
        """构建特征提取模型（支持 ResNet18/50，可选量化）"""
        backbone = self.config.MODEL_BACKBONE.lower()
        
        if backbone == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT if self.config.MODEL_PRETRAINED else None)
            expected_dim = 2048
        else:  # 默认 resnet18
            model = resnet18(weights=ResNet18_Weights.DEFAULT if self.config.MODEL_PRETRAINED else None)
            expected_dim = 512
        
        model = model.to(self.device)
        model.eval()
        
        # 移除最后一层全连接层，保留特征输出
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        
        # 应用 INT8 量化（如果启用）
        if self.config.QUANTIZATION_ENABLED and self.device.type == "cpu":
            feature_extractor = self._apply_quantization(feature_extractor)
            logger.info("已启用 INT8 量化")
        
        # 验证输出维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, *self.config.FRAME_SIZE).to(self.device)
            dummy_output = feature_extractor(dummy_input)
            actual_dim = dummy_output.shape[1]
            if actual_dim != expected_dim:
                logger.warning(f"模型输出维度不匹配: 期望 {expected_dim}, 实际 {actual_dim}")
        
        return feature_extractor

    def _apply_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """应用 INT8 动态量化"""
        try:
            # 配置量化
            model.qconfig = torch.quantization.get_default_qconfig(
                self.config.get("model.quantization.backend", "fbgemm")
            )
            # 准备量化
            torch.quantization.prepare(model, inplace=True)
            # 转换为量化模型
            torch.quantization.convert(model, inplace=True)
            logger.info("INT8 量化应用成功")
        except Exception as e:
            logger.error(f"INT8 量化失败: {e}")
        return model

    def read_frames(self, video_path: str) -> Optional[List[np.ndarray]]:
        """读取视频所有帧并转换为RGB格式"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return None

        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
        cap.release()

        if len(frames) == 0:
            logger.error("视频无有效帧")
            return None
        
        logger.debug(f"成功读取 {frame_count} 帧")
        return frames

    def sample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """等间距采样指定数量的帧"""
        if len(frames) <= self.config.SAMPLE_FRAMES:
            return frames
        
        sample_indices = np.linspace(
            0, len(frames) - 1, self.config.SAMPLE_FRAMES, dtype=int
        )
        return [frames[i] for i in sample_indices]

    def extract_frame_embedding(self, frame: np.ndarray) -> np.ndarray:
        """提取单帧的特征向量"""
        frame_tensor = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1)
        frame_tensor = self.frame_transform(frame_tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.feature_model(frame_tensor)

        return embedding.cpu().numpy().squeeze()

    def extract_video_embedding(self, video_path: str) -> Optional[List[float]]:
        """提取视频的最终特征向量（帧均值 + 标准化）"""
        logger.debug(f"开始提取视频特征: {video_path}")
        
        frames = self.read_frames(video_path)
        if frames is None:
            return None
        
        sampled_frames = self.sample_frames(frames)
        logger.debug(f"采样 {len(sampled_frames)} 帧进行处理")

        frame_embeddings = []
        for frame in sampled_frames:
            frame_emb = self.extract_frame_embedding(frame)
            frame_embeddings.append(frame_emb)

        video_emb = np.mean(frame_embeddings, axis=0)
        video_emb = normalize_embedding(video_emb)

        expected_dim = self.config.EMBEDDING_DIM
        if len(video_emb) != expected_dim:
            logger.error(f"向量维度异常（期望{expected_dim}，实际{len(video_emb)}）")
            return None

        logger.info(f"视频向量提取完成: {video_path} ({expected_dim}维)")
        return [float(v) for v in video_emb]

    def extract_batch_embeddings(self, video_paths: List[str]) -> List[Optional[List[float]]]:
        """批量提取多个视频的特征向量"""
        logger.info(f"批量处理 {len(video_paths)} 个视频")
        results = []
        for path in video_paths:
            emb = self.extract_video_embedding(path)
            results.append(emb)
        return results
