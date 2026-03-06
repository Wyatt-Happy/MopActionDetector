# action_detector.py
import numpy as np
# 导入所有需要的类型注解模块
from typing import Tuple, Optional, List
from config import MoppingDetectionConfig
from feature_extractor import VideoFeatureExtractor
from db_manager import EmbeddingDBManager


class MoppingActionDetector:
    """拖地行为检测器：基于向量相似度的行为判定"""

    def __init__(self, config: MoppingDetectionConfig = None):
        self.config = config or MoppingDetectionConfig()
        self.feature_extractor = VideoFeatureExtractor(config)
        self.db_manager = EmbeddingDBManager(config)

    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度（向量已归一化，点积=相似度）"""
        return np.dot(np.array(vec1), np.array(vec2))

    def detect(
            self,
            video_path: str,
            threshold: Optional[float] = None,
            similarity_gap: Optional[float] = None
    ) -> Tuple[bool, float, float]:
        """
        检测视频是否包含拖地行为
        返回：(是否拖地, 拖地相似度, 非拖地相似度)
        """
        # 使用配置阈值或传入阈值
        threshold = threshold or self.config.MOPPING_THRESHOLD
        similarity_gap = similarity_gap or self.config.SIMILARITY_GAP

        print(f"\n===== 【超严格模式】开始检测视频：{video_path} =====")

        # 1. 提取待测视频向量
        test_embedding = self.feature_extractor.extract_video_embedding(video_path)
        if test_embedding is None:
            print("❌ 待测视频向量提取失败")
            return False, 0.0, 0.0

        # 2. 校验数据库是否有数据
        total_count = self.db_manager.check_db_status()
        if total_count == 0:
            print("❌ 向量库无数据，无法检测")
            return False, 0.0, 0.0

        # 3. 查询相似向量（TOP1）
        mop_results = self.db_manager.query_embeddings(
            test_embedding,
            self.config.ACTION_MOPPING,
            n_results=self.config.TOP_K
        )
        non_mop_results = self.db_manager.query_embeddings(
            test_embedding,
            self.config.ACTION_NON_MOPPING,
            n_results=self.config.TOP_K
        )

        # 4. 计算相似度（变量名统一为xxx_similarity）
        mop_similarity = 0.0
        if mop_results.get("embeddings") and len(mop_results["embeddings"]) > 0 and len(
                mop_results["embeddings"][0]) > 0:
            mop_similarity = self.calculate_similarity(
                test_embedding,
                mop_results["embeddings"][0][0]
            )

        non_mop_similarity = 0.0
        if non_mop_results.get("embeddings") and len(non_mop_results["embeddings"]) > 0 and len(
                non_mop_results["embeddings"][0]) > 0:
            non_mop_similarity = self.calculate_similarity(
                test_embedding,
                non_mop_results["embeddings"][0][0]
            )

        # 5. 超严格判定规则
        is_mopping = (mop_similarity >= threshold) and (mop_similarity - non_mop_similarity >= similarity_gap)

        # 6. 打印检测结果
        print(f"\n===== 检测结果 =====")
        print(f"拖地相似度：{mop_similarity:.3f}（阈值≥{threshold}）")
        print(f"非拖地相似度：{non_mop_similarity:.3f}")
        print(f"相似度差值：{mop_similarity - non_mop_similarity:.3f}（要求≥{similarity_gap}）")
        print(f"结论：{'✅ 检测到拖地行为' if is_mopping else '❌ 未检测到拖地行为'}")

        # 核心修复：变量名统一为xxx_similarity，与定义一致
        return is_mopping, mop_similarity, non_mop_similarity