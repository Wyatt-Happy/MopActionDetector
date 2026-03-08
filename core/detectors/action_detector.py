# action_detector.py
import numpy as np
from typing import Tuple, Optional, List, Dict
from collections import deque
from core.utils.config import MoppingDetectionConfig
from core.extractors.feature_extractor import VideoFeatureExtractor
from core.managers.db_manager import EmbeddingDBManager
from core.managers.logger import get_logger

logger = get_logger(__name__)


class AdaptiveThresholdLearner:
    """自适应阈值学习器"""
    
    def __init__(self, config: MoppingDetectionConfig):
        self.config = config
        self.history = deque(maxlen=config.ADAPTIVE_WINDOW_SIZE)
        self.current_threshold = config.MOPPING_THRESHOLD
        self.current_gap = config.SIMILARITY_GAP
        
    def update(self, is_mopping: bool, mop_sim: float, non_mop_sim: float, ground_truth: bool = None):
        """更新历史记录并调整阈值"""
        if not self.config.ADAPTIVE_THRESHOLD_ENABLED:
            return
            
        self.history.append({
            'is_mopping': is_mopping,
            'mop_sim': mop_sim,
            'non_mop_sim': non_mop_sim,
            'ground_truth': ground_truth
        })
        
        # 如果有足够的历史数据且有标注，进行阈值调整
        if len(self.history) >= 10 and ground_truth is not None:
            self._adjust_threshold()
    
    def _adjust_threshold(self):
        """根据历史表现调整阈值"""
        # 计算当前阈值下的准确率
        correct = sum(1 for h in self.history if h['is_mopping'] == h['ground_truth'])
        accuracy = correct / len(self.history)
        
        lr = self.config.ADAPTIVE_LEARNING_RATE
        target = self.config.ADAPTIVE_TARGET_ACCURACY
        
        # 如果准确率低于目标，微调阈值
        if accuracy < target:
            # 根据错误类型调整
            false_positives = sum(1 for h in self.history 
                                 if h['is_mopping'] and not h['ground_truth'])
            false_negatives = sum(1 for h in self.history 
                                 if not h['is_mopping'] and h['ground_truth'])
            
            if false_positives > false_negatives:
                # 假阳性多，提高阈值
                self.current_threshold = min(0.95, self.current_threshold + lr)
            else:
                # 假阴性多，降低阈值
                self.current_threshold = max(0.5, self.current_threshold - lr)
            
            logger.info(f"自适应调整阈值: {self.current_threshold:.3f} (准确率: {accuracy:.3f})")


class MoppingActionDetector:
    """拖地行为检测器：支持多种相似度计算方法"""

    def __init__(self, config: MoppingDetectionConfig = None):
        self.config = config or MoppingDetectionConfig()
        self.feature_extractor = VideoFeatureExtractor(config)
        self.db_manager = EmbeddingDBManager(config)
        self.adaptive_learner = AdaptiveThresholdLearner(config)
        logger.info("行为检测器初始化完成")

    def calculate_similarity(self, vec1: List[float], vec2: List[float], method: str = None) -> float:
        """计算两个向量的相似度（支持多种方法）"""
        method = method or self.config.SIMILARITY_METHOD
        v1, v2 = np.array(vec1), np.array(vec2)
        
        if method == "cosine":
            # 余弦相似度（向量已归一化，点积=相似度）
            return float(np.dot(v1, v2))
        
        elif method == "weighted":
            # 加权相似度（时序+空间特征）
            weights = self.config.SIMILARITY_WEIGHTS
            # 简单实现：使用加权平均
            temporal_weight = weights.get("temporal", 0.3)
            spatial_weight = weights.get("spatial", 0.7)
            cosine_sim = np.dot(v1, v2)
            # 欧氏距离转换的相似度
            euclidean_dist = np.linalg.norm(v1 - v2)
            euclidean_sim = 1 / (1 + euclidean_dist)
            return float(temporal_weight * cosine_sim + spatial_weight * euclidean_sim)
        
        elif method == "euclidean":
            # 欧氏距离转相似度
            dist = np.linalg.norm(v1 - v2)
            return float(1 / (1 + dist))
        
        else:
            # 默认余弦相似度
            return float(np.dot(v1, v2))

    def calculate_knn_similarity(self, test_embedding: List[float], 
                                  action_type: str, k: int = 3) -> float:
        """KNN 相似度计算（取 Top-K 平均）"""
        results = self.db_manager.query_embeddings(
            test_embedding, action_type, n_results=k
        )
        
        if not results.get("embeddings") or len(results["embeddings"][0]) == 0:
            return 0.0
        
        similarities = []
        for emb in results["embeddings"][0]:
            sim = self.calculate_similarity(test_embedding, emb)
            similarities.append(sim)
        
        # 取 Top-K 相似度的平均值
        return float(np.mean(similarities)) if similarities else 0.0

    def detect(
            self,
            video_path: str,
            threshold: Optional[float] = None,
            similarity_gap: Optional[float] = None,
            ground_truth: bool = None
    ) -> Tuple[bool, float, float, Dict]:
        """
        检测视频是否包含拖地行为
        返回：(是否拖地, 拖地相似度, 非拖地相似度, 详细信息)
        """
        # 使用自适应阈值或传入阈值
        if self.config.ADAPTIVE_THRESHOLD_ENABLED and threshold is None:
            threshold = self.adaptive_learner.current_threshold
        else:
            threshold = threshold or self.config.MOPPING_THRESHOLD
        similarity_gap = similarity_gap or self.config.SIMILARITY_GAP

        logger.info(f"开始检测视频: {video_path}")

        # 1. 提取待测视频向量
        test_embedding = self.feature_extractor.extract_video_embedding(video_path)
        if test_embedding is None:
            logger.error("待测视频向量提取失败")
            return False, 0.0, 0.0, {"error": "向量提取失败"}

        # 2. 校验数据库
        total_count = self.db_manager.check_db_status()
        if total_count == 0:
            logger.error("向量库无数据，无法检测")
            return False, 0.0, 0.0, {"error": "数据库为空"}

        # 3. 计算相似度
        method = self.config.SIMILARITY_METHOD
        if method == "knn":
            mop_similarity = self.calculate_knn_similarity(
                test_embedding, self.config.ACTION_MOPPING, k=self.config.TOP_K
            )
            non_mop_similarity = self.calculate_knn_similarity(
                test_embedding, self.config.ACTION_NON_MOPPING, k=self.config.TOP_K
            )
        else:
            # 查询相似向量
            mop_results = self.db_manager.query_embeddings(
                test_embedding, self.config.ACTION_MOPPING, n_results=self.config.TOP_K
            )
            non_mop_results = self.db_manager.query_embeddings(
                test_embedding, self.config.ACTION_NON_MOPPING, n_results=self.config.TOP_K
            )
            
            # 计算相似度
            mop_similarity = 0.0
            if mop_results.get("embeddings") and len(mop_results["embeddings"]) > 0:
                mop_similarity = self.calculate_similarity(
                    test_embedding, mop_results["embeddings"][0][0], method
                )
            
            non_mop_similarity = 0.0
            if non_mop_results.get("embeddings") and len(non_mop_results["embeddings"]) > 0:
                non_mop_similarity = self.calculate_similarity(
                    test_embedding, non_mop_results["embeddings"][0][0], method
                )

        # 4. 判定规则
        is_mopping = (mop_similarity >= threshold) and (mop_similarity - non_mop_similarity >= similarity_gap)

        # 5. 更新自适应阈值
        self.adaptive_learner.update(is_mopping, mop_similarity, non_mop_similarity, ground_truth)

        # 6. 构建详细信息
        details = {
            "method": method,
            "threshold": threshold,
            "similarity_gap": similarity_gap,
            "mop_similarity": round(mop_similarity, 4),
            "non_mop_similarity": round(non_mop_similarity, 4),
            "similarity_diff": round(mop_similarity - non_mop_similarity, 4),
            "confidence": round(abs(mop_similarity - non_mop_similarity), 4)
        }

        logger.info(f"检测完成: {video_path} - {'拖地' if is_mopping else '非拖地'} "
                   f"(拖地相似度: {mop_similarity:.3f})")

        return is_mopping, mop_similarity, non_mop_similarity, details

    def detect_batch(self, video_paths: List[str]) -> List[Tuple[bool, float, float, Dict]]:
        """批量检测多个视频"""
        logger.info(f"批量检测 {len(video_paths)} 个视频")
        results = []
        for path in video_paths:
            result = self.detect(path)
            results.append(result)
        return results
