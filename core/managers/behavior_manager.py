# behavior_manager.py
"""
动态行为管理器
支持通过UI创建新行为，自动管理ChromaDB集合
"""

import os
import chromadb
from typing import List, Dict, Optional, Any
from datetime import datetime
from core.utils.config import MoppingDetectionConfig
from core.managers.logger import get_logger

logger = get_logger(__name__)


class BehaviorManager:
    """
    动态行为管理器
    
    功能：
    - 创建/删除行为类型
    - 管理每个行为的ChromaDB集合
    - 记录行为元数据
    """
    
    def __init__(self, config: MoppingDetectionConfig = None):
        self.config = config or MoppingDetectionConfig()
        self.client = chromadb.PersistentClient(path=self.config.DB_PATH)
        self.behaviors_collection = self._init_behaviors_metadata_collection()
        
    def _init_behaviors_metadata_collection(self):
        """初始化行为元数据集合"""
        try:
            collection = self.client.get_or_create_collection(
                name="behavior_metadata",
                metadata={"description": "存储所有行为的元数据信息"}
            )
            logger.info("行为元数据集合初始化完成")
            return collection
        except Exception as e:
            logger.error(f"行为元数据集合初始化失败: {e}")
            raise
    
    def create_behavior(
        self,
        behavior_id: str,
        display_name: str,
        category: str = "general",
        description: str = "",
        color: str = "#667eea"
    ) -> Dict[str, Any]:
        """
        创建新行为类型
        
        Args:
            behavior_id: 行为唯一标识（如：mopping, cashier）
            display_name: 显示名称（如：拖地, 收银）
            category: 分类（hygiene/service/safety/violation/general）
            description: 行为描述
            color: UI显示颜色
            
        Returns:
            创建的行为信息
        """
        try:
            # 检查行为ID是否已存在
            existing = self.get_behavior(behavior_id)
            if existing:
                logger.warning(f"行为 {behavior_id} 已存在")
                return existing
            
            # 创建ChromaDB集合
            collection_name = f"behavior_{behavior_id}"
            self.client.create_collection(
                name=collection_name,
                metadata={
                    "behavior_id": behavior_id,
                    "display_name": display_name,
                    "created_at": datetime.now().isoformat()
                }
            )
            
            # 记录行为元数据
            behavior_info = {
                "behavior_id": behavior_id,
                "display_name": display_name,
                "category": category,
                "description": description,
                "color": color,
                "collection_name": collection_name,
                "video_count": 0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            self.behaviors_collection.add(
                ids=[behavior_id],
                documents=[description],
                metadatas=[behavior_info]
            )
            
            logger.info(f"成功创建行为: {behavior_id} ({display_name})")
            return behavior_info
            
        except Exception as e:
            logger.error(f"创建行为失败: {e}")
            raise
    
    def delete_behavior(self, behavior_id: str) -> bool:
        """
        删除行为类型
        
        Args:
            behavior_id: 行为ID
            
        Returns:
            是否删除成功
        """
        try:
            # 删除ChromaDB集合
            collection_name = f"behavior_{behavior_id}"
            try:
                self.client.delete_collection(name=collection_name)
            except:
                pass
            
            # 删除元数据
            self.behaviors_collection.delete(ids=[behavior_id])
            
            logger.info(f"成功删除行为: {behavior_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除行为失败: {e}")
            return False
    
    def get_behavior(self, behavior_id: str) -> Optional[Dict[str, Any]]:
        """
        获取行为信息
        
        Args:
            behavior_id: 行为ID
            
        Returns:
            行为信息，不存在返回None
        """
        try:
            result = self.behaviors_collection.get(ids=[behavior_id])
            if result and result["metadatas"]:
                return result["metadatas"][0]
            return None
        except:
            return None
    
    def list_behaviors(self, category: str = None) -> List[Dict[str, Any]]:
        """
        列出所有行为
        
        Args:
            category: 按分类筛选，None返回所有
            
        Returns:
            行为列表
        """
        try:
            result = self.behaviors_collection.get()
            behaviors = []
            
            if result and result["metadatas"]:
                for metadata in result["metadatas"]:
                    if category is None or metadata.get("category") == category:
                        # 获取实际视频数量
                        video_count = self._get_behavior_video_count(
                            metadata["behavior_id"]
                        )
                        metadata["video_count"] = video_count
                        behaviors.append(metadata)
            
            return behaviors
            
        except Exception as e:
            logger.error(f"获取行为列表失败: {e}")
            return []
    
    def _get_behavior_video_count(self, behavior_id: str) -> int:
        """获取行为的视频数量"""
        try:
            collection_name = f"behavior_{behavior_id}"
            collection = self.client.get_collection(name=collection_name)
            return collection.count()
        except:
            return 0
    
    def update_behavior(
        self,
        behavior_id: str,
        display_name: str = None,
        description: str = None,
        color: str = None
    ) -> bool:
        """
        更新行为信息
        
        Args:
            behavior_id: 行为ID
            display_name: 新显示名称
            description: 新描述
            color: 新颜色
            
        Returns:
            是否更新成功
        """
        try:
            behavior = self.get_behavior(behavior_id)
            if not behavior:
                logger.warning(f"行为 {behavior_id} 不存在")
                return False
            
            # 更新字段
            if display_name:
                behavior["display_name"] = display_name
            if description:
                behavior["description"] = description
            if color:
                behavior["color"] = color
            
            behavior["updated_at"] = datetime.now().isoformat()
            
            # 保存更新
            self.behaviors_collection.update(
                ids=[behavior_id],
                documents=[behavior.get("description", "")],
                metadatas=[behavior]
            )
            
            logger.info(f"成功更新行为: {behavior_id}")
            return True
            
        except Exception as e:
            logger.error(f"更新行为失败: {e}")
            return False
    
    def get_collection(self, behavior_id: str):
        """
        获取行为的ChromaDB集合
        
        Args:
            behavior_id: 行为ID
            
        Returns:
            ChromaDB集合对象
        """
        collection_name = f"behavior_{behavior_id}"
        return self.client.get_collection(name=collection_name)
    
    def get_all_collections(self) -> List[str]:
        """
        获取所有行为集合名称
        
        Returns:
            集合名称列表
        """
        try:
            collections = self.client.list_collections()
            return [
                c.name for c in collections 
                if c.name.startswith("behavior_") and c.name != "behavior_metadata"
            ]
        except Exception as e:
            logger.error(f"获取集合列表失败: {e}")
            return []


# 预定义的行为分类
BEHAVIOR_CATEGORIES = {
    "hygiene": {
        "name": "卫生行为",
        "color": "#28a745",
        "behaviors": ["mopping", "cleaning", "hand_washing"]
    },
    "service": {
        "name": "服务行为", 
        "color": "#007bff",
        "behaviors": ["cashier", "stocking", "customer_service"]
    },
    "safety": {
        "name": "安全行为",
        "color": "#ffc107", 
        "behaviors": ["wearing_gloves", "wearing_mask"]
    },
    "violation": {
        "name": "违规行为",
        "color": "#dc3545",
        "behaviors": ["phone_usage", "smoking", "eating"]
    },
    "general": {
        "name": "一般行为",
        "color": "#6c757d",
        "behaviors": []
    }
}
