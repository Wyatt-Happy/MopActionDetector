# config.py
import os
import yaml
import torch
from typing import Dict, Any, Tuple, List
from pathlib import Path


class MoppingDetectionConfig:
    """拖地检测全局配置类（支持 YAML 配置文件）"""
    
    _instance = None
    _config_data = None
    
    def __new__(cls, config_path: str = None):
        """单例模式，确保配置只加载一次"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: str = None):
        if self._initialized:
            return
            
        # 默认配置文件路径
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "config.yaml")
        
        # 加载 YAML 配置
        self._config_path = config_path
        self._load_config()
        
        # 初始化派生属性
        self._init_derived_properties()
        
        self._initialized = True
    
    def _load_config(self):
        """从 YAML 文件加载配置"""
        if not os.path.exists(self._config_path):
            print(f"⚠️ 配置文件不存在: {self._config_path}，使用默认配置")
            self._config_data = self._get_default_config()
        else:
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    self._config_data = yaml.safe_load(f)
                print(f"✅ 配置文件加载成功: {self._config_path}")
            except Exception as e:
                print(f"❌ 配置文件加载失败: {e}，使用默认配置")
                self._config_data = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "device": {"type": "auto", "gpu_id": 0},
            "model": {"backbone": "resnet18", "pretrained": True, "embedding_dim": 512},
            "video": {
                "frame_size": [224, 224],
                "sample_frames": 32,
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225]
            },
            "database": {
                "path": "E:/PoseYOLO/video_embedding_db",
                "collection_name": "action_video_embeddings"
            },
            "detection": {
                "mopping_threshold": 0.75,
                "similarity_gap": 0.1,
                "top_k": 1,
                "similarity_method": "cosine"
            },
            "logging": {"level": "INFO", "file": "logs/mopping_detection.log"},
            "actions": {"mopping": "mopping", "non_mopping": "non_mopping"}
        }
    
    def _init_derived_properties(self):
        """初始化派生属性"""
        # 设备配置
        device_type = self._get_nested_value("device.type", "auto")
        if device_type == "auto":
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.DEVICE = device_type
        self.GPU_ID = self._get_nested_value("device.gpu_id", 0)
        
        # 模型配置
        self.MODEL_BACKBONE = self._get_nested_value("model.backbone", "resnet18")
        self.MODEL_PRETRAINED = self._get_nested_value("model.pretrained", True)
        self.EMBEDDING_DIM = self._get_nested_value("model.embedding_dim", 512)
        self.QUANTIZATION_ENABLED = self._get_nested_value("model.quantization.enabled", False)
        
        # 视频处理配置
        frame_size = self._get_nested_value("video.frame_size", [224, 224])
        self.FRAME_SIZE = tuple(frame_size)
        self.SAMPLE_FRAMES = self._get_nested_value("video.sample_frames", 32)
        mean = self._get_nested_value("video.normalize_mean", [0.485, 0.456, 0.406])
        std = self._get_nested_value("video.normalize_std", [0.229, 0.224, 0.225])
        self.NORMALIZE_MEAN = tuple(mean)
        self.NORMALIZE_STD = tuple(std)
        self.SUPPORTED_FORMATS = tuple(self._get_nested_value("video.supported_formats", [".mp4", ".avi", ".mov", ".mkv"]))
        
        # 实时流配置
        self.STREAM_CAMERA_ID = self._get_nested_value("stream.camera_id", 0)
        self.STREAM_BUFFER_SIZE = self._get_nested_value("stream.buffer_size", 10)
        self.STREAM_SAMPLE_INTERVAL = self._get_nested_value("stream.sample_interval", 2)
        self.STREAM_SHOW_PREVIEW = self._get_nested_value("stream.show_preview", True)
        
        # 数据库配置
        self.DB_PATH = self._get_nested_value("database.path", "E:/PoseYOLO/video_embedding_db")
        self.COLLECTION_NAME = self._get_nested_value("database.collection_name", "action_video_embeddings")
        self.COLLECTION_METADATA = self._get_nested_value("database.metadata", {"description": "拖地/非拖地行为视频特征向量库"})
        
        # 检测配置
        self.MOPPING_THRESHOLD = self._get_nested_value("detection.mopping_threshold", 0.75)
        self.SIMILARITY_GAP = self._get_nested_value("detection.similarity_gap", 0.1)
        self.TOP_K = self._get_nested_value("detection.top_k", 1)
        self.SIMILARITY_METHOD = self._get_nested_value("detection.similarity_method", "cosine")
        self.SIMILARITY_WEIGHTS = self._get_nested_value("detection.weights", {"temporal": 0.3, "spatial": 0.7})
        
        # 自适应阈值配置
        self.ADAPTIVE_THRESHOLD_ENABLED = self._get_nested_value("adaptive_threshold.enabled", False)
        self.ADAPTIVE_LEARNING_RATE = self._get_nested_value("adaptive_threshold.learning_rate", 0.01)
        self.ADAPTIVE_WINDOW_SIZE = self._get_nested_value("adaptive_threshold.window_size", 100)
        self.ADAPTIVE_TARGET_ACCURACY = self._get_nested_value("adaptive_threshold.target_accuracy", 0.95)
        
        # 日志配置
        self.LOG_LEVEL = self._get_nested_value("logging.level", "INFO")
        self.LOG_FILE = self._get_nested_value("logging.file", "logs/mopping_detection.log")
        self.LOG_FORMAT = self._get_nested_value("logging.format", 
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.LOG_CONSOLE = self._get_nested_value("logging.console", True)
        self.LOG_ROTATION_ENABLED = self._get_nested_value("logging.rotation.enabled", True)
        self.LOG_ROTATION_MAX_BYTES = self._get_nested_value("logging.rotation.max_bytes", 10485760)
        self.LOG_ROTATION_BACKUP_COUNT = self._get_nested_value("logging.rotation.backup_count", 5)
        
        # 导出配置
        self.EXPORT_FORMAT = self._get_nested_value("export.format", "csv")
        self.EXPORT_OUTPUT_DIR = self._get_nested_value("export.output_dir", "exports")
        self.EXPORT_INCLUDE_TIMESTAMP = self._get_nested_value("export.include_timestamp", True)
        
        # 并行处理配置
        self.PARALLEL_ENABLED = self._get_nested_value("parallel.enabled", True)
        self.PARALLEL_WORKERS = self._get_nested_value("parallel.workers", 0)
        self.PARALLEL_QUEUE_SIZE = self._get_nested_value("parallel.queue_size", 100)
        self.PARALLEL_BATCH_SIZE = self._get_nested_value("parallel.batch_size", 4)
        
        # API 配置
        self.API_HOST = self._get_nested_value("api.host", "0.0.0.0")
        self.API_PORT = self._get_nested_value("api.port", 8000)
        self.API_DEBUG = self._get_nested_value("api.debug", False)
        self.API_MAX_FILE_SIZE = self._get_nested_value("api.max_file_size", 100)
        self.API_CORS_ENABLED = self._get_nested_value("api.cors.enabled", True)
        self.API_CORS_ORIGINS = self._get_nested_value("api.cors.origins", ["*"])
        self.API_WEB_UI_ENABLED = self._get_nested_value("api.web_ui.enabled", True)
        self.API_WEB_UI_TITLE = self._get_nested_value("api.web_ui.title", "拖地行为检测系统")
        
        # 动作类型常量
        self.ACTION_MOPPING = self._get_nested_value("actions.mopping", "mopping")
        self.ACTION_NON_MOPPING = self._get_nested_value("actions.non_mopping", "non_mopping")
    
    def _get_nested_value(self, key_path: str, default: Any = None) -> Any:
        """获取嵌套配置值"""
        keys = key_path.split(".")
        value = self._config_data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值（支持嵌套键）"""
        return self._get_nested_value(key_path, default)
    
    def reload(self):
        """重新加载配置"""
        self._load_config()
        self._init_derived_properties()
        print("✅ 配置已重新加载")
    
    def to_dict(self) -> Dict[str, Any]:
        """导出配置为字典"""
        return {
            "device": self.DEVICE,
            "model_backbone": self.MODEL_BACKBONE,
            "frame_size": self.FRAME_SIZE,
            "sample_frames": self.SAMPLE_FRAMES,
            "db_path": self.DB_PATH,
            "mopping_threshold": self.MOPPING_THRESHOLD,
            "similarity_gap": self.SIMILARITY_GAP,
            "top_k": self.TOP_K
        }
