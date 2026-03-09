# temporal_extractor.py
"""
时序特征提取器
使用 LSTM 捕获视频的时序信息
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from core.utils.config import MoppingDetectionConfig
from core.managers.logger import get_logger

logger = get_logger(__name__)


class TemporalFeatureExtractor:
    """
    时序特征提取器
    
    使用 LSTM 对帧级特征进行时序建模，捕获动作的时间序列信息
    """
    
    def __init__(
        self,
        config: MoppingDetectionConfig = None,
        input_size: int = 512,  # ResNet18 特征维度
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.5
    ):
        """
        初始化时序特征提取器
        
        Args:
            config: 配置对象
            input_size: 输入特征维度
            hidden_size: LSTM 隐藏层维度
            num_layers: LSTM 层数
            bidirectional: 是否使用双向 LSTM
            dropout:  dropout 概率
        """
        self.config = config or MoppingDetectionConfig()
        self.device = self._get_device()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 计算输出特征维度
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # 构建 LSTM 模型
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 移动到设备
        self.lstm = self.lstm.to(self.device)
        self.lstm.eval()
        
        logger.info(
            f"时序特征提取器初始化完成: "
            f"输入维度={input_size}, 隐藏维度={hidden_size}, "
            f"层数={num_layers}, 双向={bidirectional}"
        )
    
    def _get_device(self) -> torch.device:
        """获取计算设备"""
        device_type = self.config.DEVICE
        
        if device_type == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("使用 CPU")
        
        return device
    
    def extract(
        self,
        frame_features: List[np.ndarray]
    ) -> np.ndarray:
        """
        提取时序特征
        
        Args:
            frame_features: 帧级特征列表，每个元素是 (feature_dim,) 的 numpy 数组
            
        Returns:
            时序特征 (output_size,)
        """
        try:
            if not frame_features:
                raise ValueError("帧特征列表为空")
            
            # 转换为张量
            features_tensor = torch.tensor(
                np.stack(frame_features),
                dtype=torch.float32,
                device=self.device
            )
            
            # 添加 batch 维度
            features_tensor = features_tensor.unsqueeze(0)  # (1, seq_len, feature_dim)
            
            # 提取时序特征
            with torch.no_grad():
                output, (h_n, c_n) = self.lstm(features_tensor)
            
            # 获取最后一层的隐藏状态
            if self.bidirectional:
                # 双向 LSTM：拼接两个方向的最后隐藏状态
                h_fwd = h_n[-2, :, :]  # 正向 LSTM 最后隐藏状态
                h_bwd = h_n[-1, :, :]  # 反向 LSTM 最后隐藏状态
                temporal_feature = torch.cat([h_fwd, h_bwd], dim=1)
            else:
                # 单向 LSTM：取最后一层的最后隐藏状态
                temporal_feature = h_n[-1, :, :]
            
            # 转换为 numpy 数组并归一化
            temporal_feature = temporal_feature.squeeze().cpu().numpy()
            temporal_feature = temporal_feature / (np.linalg.norm(temporal_feature) + 1e-8)
            
            return temporal_feature.astype(np.float32)
            
        except Exception as e:
            logger.error(f"时序特征提取失败: {e}")
            raise
    
    def extract_batch(
        self,
        batch_frame_features: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """
        批量提取时序特征
        
        Args:
            batch_frame_features: 批量帧特征列表
            
        Returns:
            时序特征列表
        """
        features = []
        for frame_features in batch_frame_features:
            feature = self.extract(frame_features)
            features.append(feature)
        return features
    
    def get_output_dim(self) -> int:
        """获取输出特征维度"""
        return self.output_size
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "state_dict": self.lstm.state_dict()
        }, path)
        logger.info(f"时序模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.input_size = checkpoint["input_size"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.bidirectional = checkpoint["bidirectional"]
        
        # 重建 LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        
        self.lstm.load_state_dict(checkpoint["state_dict"])
        self.lstm = self.lstm.to(self.device)
        self.lstm.eval()
        
        # 重新计算输出维度
        self.output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        logger.info(f"时序模型已从: {path} 加载")
