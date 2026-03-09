# universal_extractor.py
"""
通用特征提取器
支持多种backbone，统一提取视频特征
"""

import cv2
import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any
from torchvision.transforms import Compose, Lambda, Resize, Normalize
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from core.utils.config import MoppingDetectionConfig
from core.managers.logger import get_logger

logger = get_logger(__name__)


class UniversalFeatureExtractor:
    """
    通用视频特征提取器
    
    支持：
    - 多种backbone（ResNet18/50/101）
    - 可配置的输入尺寸和采样帧数
    - 自动设备选择（CPU/CUDA）
    - 可选INT8量化
    """
    
    def __init__(
        self,
        config: MoppingDetectionConfig = None,
        backbone: str = "resnet18",
        input_size: int = 224,
        sample_frames: int = 32,
        pretrained: bool = True,
        quantization: bool = False
    ):
        """
        初始化通用特征提取器
        
        Args:
            config: 配置对象
            backbone: 骨干网络（resnet18/resnet50）
            input_size: 输入图像尺寸
            sample_frames: 采样帧数
            pretrained: 是否使用预训练权重
            quantization: 是否启用INT8量化
        """
        self.config = config or MoppingDetectionConfig()
        self.backbone_name = backbone
        self.input_size = input_size
        self.sample_frames = sample_frames
        self.device = self._get_device()
        
        # 加载模型
        self.model = self._load_model(pretrained)
        
        # 应用量化（如果启用）
        if quantization:
            self.model = self._apply_quantization()
        
        # 定义预处理
        self.transform = self._build_transform()
        
        logger.info(
            f"通用特征提取器初始化完成: {backbone}, "
            f"输入尺寸: {input_size}, 采样帧数: {sample_frames}"
        )
    
    def _get_device(self) -> torch.device:
        """获取计算设备"""
        if self.config.DEVICE == "cuda" and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config.GPU_ID}")
            logger.info(f"使用GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            logger.info("使用CPU")
        return device
    
    def _load_model(self, pretrained: bool) -> torch.nn.Module:
        """加载骨干网络"""
        try:
            if self.backbone_name == "resnet18":
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                model = resnet18(weights=weights)
                self.feature_dim = 512
            elif self.backbone_name == "resnet50":
                weights = ResNet50_Weights.DEFAULT if pretrained else None
                model = resnet50(weights=weights)
                self.feature_dim = 2048
            else:
                raise ValueError(f"不支持的backbone: {self.backbone_name}")
            
            # 移除分类层，保留特征提取层
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _apply_quantization(self) -> torch.nn.Module:
        """应用INT8量化"""
        try:
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self.model, inplace=True)
            torch.quantization.convert(self.model, inplace=True)
            logger.info("INT8量化已启用")
            return self.model
        except Exception as e:
            logger.warning(f"量化失败: {e}，使用原始模型")
            return self.model
    
    def _build_transform(self) -> Compose:
        """构建图像预处理流程"""
        return Compose([
            Lambda(lambda x: x / 255.0),
            Resize((self.input_size, self.input_size)),
            Normalize(
                mean=self.config.NORMALIZE_MEAN,
                std=self.config.NORMALIZE_STD
            )
        ])
    
    def extract_features(
        self,
        video_path: str,
        task_type: str = "general",
        return_frames: bool = False
    ) -> np.ndarray:
        """
        提取视频特征
        
        Args:
            video_path: 视频文件路径
            task_type: 任务类型（general/hygiene/service/safety）
            return_frames: 是否返回帧级特征列表
            
        Returns:
            特征向量 (feature_dim,) 或帧级特征列表
        """
        try:
            # 读取视频帧
            frames = self._read_video_frames(video_path)
            if len(frames) == 0:
                raise ValueError(f"无法读取视频: {video_path}")
            
            # 采样帧
            sampled_frames = self._sample_frames(frames)
            
            # 根据任务类型调整采样策略
            if task_type == "hygiene":
                # 卫生行为：关注动作细节，增加采样密度
                sampled_frames = self._enhance_motion_sampling(sampled_frames)
            elif task_type == "service":
                # 服务行为：关注交互，保持均匀采样
                pass
            
            # 提取每帧特征
            features = []
            with torch.no_grad():
                for frame in sampled_frames:
                    # 预处理
                    frame_tensor = self._preprocess_frame(frame)
                    frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
                    
                    # 提取特征
                    feature = self.model(frame_tensor)
                    feature = feature.squeeze().cpu().numpy()
                    features.append(feature)
            
            if return_frames:
                return features
            
            # 聚合特征（取平均）
            video_feature = np.mean(features, axis=0)
            
            # L2归一化
            video_feature = video_feature / (np.linalg.norm(video_feature) + 1e-8)
            
            return video_feature.astype(np.float32)
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            raise
    
    def extract_features_batch(
        self,
        video_paths: List[str],
        task_type: str = "general"
    ) -> List[np.ndarray]:
        """
        批量提取视频特征
        
        Args:
            video_paths: 视频路径列表
            task_type: 任务类型
            
        Returns:
            特征向量列表
        """
        features = []
        for path in video_paths:
            try:
                feature = self.extract_features(path, task_type)
                features.append(feature)
            except Exception as e:
                logger.error(f"提取 {path} 失败: {e}")
                features.append(None)
        return features
    
    def _read_video_frames(self, video_path: str) -> List[np.ndarray]:
        """读取视频所有帧"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _sample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """等间距采样帧"""
        total_frames = len(frames)
        if total_frames <= self.sample_frames:
            return frames
        
        indices = np.linspace(0, total_frames - 1, self.sample_frames, dtype=int)
        return [frames[i] for i in indices]
    
    def _enhance_motion_sampling(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """增强动作采样（用于卫生行为等需要关注动作细节的场景）"""
        # 可以在这里实现更复杂的采样策略
        # 例如：基于光流检测关键帧
        return frames
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """预处理单帧图像"""
        # numpy (H, W, C) -> tensor (C, H, W)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        frame_tensor = self.transform(frame_tensor)
        return frame_tensor
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return self.feature_dim
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "backbone": self.backbone_name,
            "input_size": self.input_size,
            "sample_frames": self.sample_frames,
            "feature_dim": self.feature_dim,
            "device": str(self.device),
            "quantization": hasattr(self.model, 'qconfig')
        }
