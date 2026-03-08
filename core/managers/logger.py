# logger.py
import os
import logging
import logging.handlers
from typing import Optional
from core.utils.config import MoppingDetectionConfig


class LoggerManager:
    """日志管理器（单例模式）"""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.config = MoppingDetectionConfig()
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志系统"""
        # 创建日志目录
        log_file = self.config.LOG_FILE
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.LOG_LEVEL.upper(), logging.INFO))
        
        # 清除现有处理器
        root_logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(self.config.LOG_FORMAT)
        
        # 控制台处理器
        if self.config.LOG_CONSOLE:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # 文件处理器（支持轮转）
        if self.config.LOG_ROTATION_ENABLED:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.LOG_ROTATION_MAX_BYTES,
                backupCount=self.config.LOG_ROTATION_BACKUP_COUNT,
                encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        file_handler.setLevel(getattr(logging, self.config.LOG_LEVEL.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的日志记录器"""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
        return self._loggers[name]


# 全局日志管理器实例
logger_manager = LoggerManager()


def get_logger(name: str) -> logging.Logger:
    """获取日志记录器的便捷函数"""
    return logger_manager.get_logger(name)
