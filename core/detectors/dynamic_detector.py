# dynamic_detector.py
"""
动态行为检测器
支持动态添加行为类型，基于相似度匹配进行检测
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from core.utils.config import MoppingDetectionConfig
from core.extractors.universal_extractor import UniversalFeatureExtractor
from core.managers.behavior_manager import BehaviorManager
from core.managers.logger import get_logger

logger = get_logger(__name__)


class DynamicBehaviorDetector:
    """
    动态行为检测器
    
    核心功能：
    - 与所有已定义行为进行相似度比对
    - 返回最匹配的行为类型及置信度
    - 支持动态添加新行为
    """
    
    def __init__(self, config: MoppingDetectionConfig = None):
        self.config = config or MoppingDetectionConfig()
        self.extractor = UniversalFeatureExtractor(config)
        self.behavior_manager = BehaviorManager(config)
        
        # 检测阈值
        self.similarity_threshold = 0.6  # 最低相似度阈值
        self.top_k = 3  # 返回前K个结果
        
        logger.info("动态行为检测器初始化完成")
    
    def detect(
        self,
        video_path: str,
        return_all: bool = False
    ) -> Dict[str, Any]:
        """
        检测视频中的行为
        
        Args:
            video_path: 视频路径
            return_all: 是否返回所有行为的相似度
            
        Returns:
            检测结果
            {
                "behavior_id": "mopping",
                "behavior_name": "拖地",
                "confidence": 0.92,
                "all_results": [...],  # 如果return_all=True
                "conclusion": "该视频最可能是：拖地（置信度92%）"
            }
        """
        try:
            # 1. 提取视频特征
            logger.info(f"开始检测视频: {video_path}")
            query_feature = self.extractor.extract_features(video_path)
            
            # 2. 获取所有行为
            behaviors = self.behavior_manager.list_behaviors()
            if not behaviors:
                return {
                    "error": "系统中没有定义任何行为，请先创建行为类型"
                }
            
            # 3. 与每个行为比对
            results = []
            for behavior in behaviors:
                behavior_id = behavior["behavior_id"]
                
                # 查询该行为数据库中最相似的视频
                similarity = self._query_behavior_similarity(
                    behavior_id, query_feature
                )
                
                results.append({
                    "behavior_id": behavior_id,
                    "behavior_name": behavior["display_name"],
                    "category": behavior.get("category", "general"),
                    "similarity": float(similarity),
                    "color": behavior.get("color", "#667eea")
                })
            
            # 4. 按相似度排序
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # 5. 构建返回结果
            top_result = results[0]
            
            detection_result = {
                "behavior_id": top_result["behavior_id"],
                "behavior_name": top_result["behavior_name"],
                "category": top_result["category"],
                "confidence": top_result["similarity"],
                "threshold": self.similarity_threshold,
                "is_confident": top_result["similarity"] >= self.similarity_threshold,
                "timestamp": datetime.now().isoformat()
            }
            
            # 生成结论
            if detection_result["is_confident"]:
                detection_result["conclusion"] = (
                    f"该视频最可能是：{top_result['behavior_name']} "
                    f"（置信度{top_result['similarity']:.1%}）"
                )
            else:
                detection_result["conclusion"] = (
                    f"无法确定行为类型，最可能是：{top_result['behavior_name']} "
                    f"（置信度{top_result['similarity']:.1%}，低于阈值{self.similarity_threshold}）"
                )
            
            # 6. 如果需要，返回所有结果
            if return_all:
                detection_result["all_results"] = results[:self.top_k]
            
            logger.info(
                f"检测完成: {top_result['behavior_name']} "
                f"({top_result['similarity']:.3f})"
            )
            
            return detection_result
            
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return {"error": str(e)}
    
    def detect_batch(
        self,
        video_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """
        批量检测视频
        
        Args:
            video_paths: 视频路径列表
            
        Returns:
            检测结果列表
        """
        results = []
        for path in video_paths:
            result = self.detect(path)
            result["video_path"] = path
            results.append(result)
        return results
    
    def _query_behavior_similarity(
        self,
        behavior_id: str,
        query_feature: np.ndarray
    ) -> float:
        """
        查询与指定行为的相似度
        
        Args:
            behavior_id: 行为ID
            query_feature: 查询特征向量
            
        Returns:
            最高相似度分数
        """
        try:
            collection = self.behavior_manager.get_collection(behavior_id)
            
            # 查询最相似的1个结果
            results = collection.query(
                query_embeddings=[query_feature.tolist()],
                n_results=1
            )
            
            if results and results["distances"] and results["distances"][0]:
                # ChromaDB返回的是距离，转换为相似度
                distance = results["distances"][0][0]
                similarity = 1 - distance  # 简单转换
                return max(0.0, min(1.0, similarity))
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"查询行为 {behavior_id} 失败: {e}")
            return 0.0
    
    def add_training_video(
        self,
        behavior_id: str,
        video_path: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        为指定行为添加训练视频
        
        Args:
            behavior_id: 行为ID
            video_path: 视频路径
            metadata: 额外元数据
            
        Returns:
            是否添加成功
        """
        try:
            # 1. 检查行为是否存在
            behavior = self.behavior_manager.get_behavior(behavior_id)
            if not behavior:
                logger.error(f"行为 {behavior_id} 不存在")
                return False
            
            # 2. 提取特征
            feature = self.extractor.extract_features(video_path)
            
            # 3. 生成唯一ID
            video_id = f"{behavior_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 4. 准备元数据
            if metadata is None:
                metadata = {}
            metadata.update({
                "behavior_id": behavior_id,
                "source_video": video_path,
                "added_at": datetime.now().isoformat()
            })
            
            # 5. 添加到数据库
            collection = self.behavior_manager.get_collection(behavior_id)
            collection.add(
                ids=[video_id],
                embeddings=[feature.tolist()],
                metadatas=[metadata]
            )
            
            logger.info(f"成功添加训练视频到 {behavior_id}: {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"添加训练视频失败: {e}")
            return False
    
    def get_behavior_stats(self) -> Dict[str, Any]:
        """
        获取所有行为的统计信息
        
        Returns:
            统计信息
        """
        behaviors = self.behavior_manager.list_behaviors()
        
        stats = {
            "total_behaviors": len(behaviors),
            "total_videos": sum(b.get("video_count", 0) for b in behaviors),
            "behaviors": behaviors,
            "categories": {}
        }
        
        # 按分类统计
        for behavior in behaviors:
            category = behavior.get("category", "general")
            if category not in stats["categories"]:
                stats["categories"][category] = {
                    "count": 0,
                    "videos": 0
                }
            stats["categories"][category]["count"] += 1
            stats["categories"][category]["videos"] += behavior.get("video_count", 0)
        
        return stats
    
    def set_threshold(self, threshold: float):
        """
        设置检测阈值
        
        Args:
            threshold: 相似度阈值（0-1）
        """
        self.similarity_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"检测阈值设置为: {self.similarity_threshold}")
