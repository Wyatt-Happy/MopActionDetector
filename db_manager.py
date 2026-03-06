# db_manager.py
import chromadb
import traceback
import numpy as np
from typing import List, Union, Optional
from chromadb.api.models.Collection import Collection
from config import MoppingDetectionConfig
from feature_extractor import VideoFeatureExtractor
from utils import validate_video_path, generate_unique_id


class EmbeddingDBManager:
    """向量数据库管理器：适配新版ChromaDB的删除逻辑"""

    def __init__(self, config: MoppingDetectionConfig = None):
        self.config = config or MoppingDetectionConfig()
        self.client = self._init_db_client()
        self.collection = self._get_or_create_collection()
        self.feature_extractor = VideoFeatureExtractor(config)

    def _init_db_client(self) -> chromadb.PersistentClient:
        """初始化ChromaDB客户端（持久化存储）"""
        return chromadb.PersistentClient(path=self.config.DB_PATH)

    def _get_or_create_collection(self) -> Collection:
        """获取或创建向量集合"""
        return self.client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            metadata=self.config.COLLECTION_METADATA
        )

    def add_video_embeddings(
            self,
            video_paths: Union[str, List[str]],
            action_type: str
    ) -> int:
        """批量添加视频向量到数据库"""
        # 1. 校验输入
        if not validate_video_path(video_paths):
            return 0
        if action_type not in [self.config.ACTION_MOPPING, self.config.ACTION_NON_MOPPING]:
            print(f"❌ 无效动作类型：{action_type}（仅支持{self.config.ACTION_MOPPING}/{self.config.ACTION_NON_MOPPING}）")
            return 0

        # 标准化为列表
        if isinstance(video_paths, str):
            video_paths = [video_paths]

        success_count = 0
        print(f"\n===== 开始入库 {action_type} 视频（共{len(video_paths)}个） =====")

        # 2. 逐个提取并入库
        for idx, video_path in enumerate(video_paths):
            print(f"\n【{idx + 1}/{len(video_paths)}】处理：{video_path}")
            # 提取向量
            embedding = self.feature_extractor.extract_video_embedding(video_path)
            if embedding is None:
                print(f"❌ 跳过：{video_path} 向量提取失败")
                continue

            # 生成唯一ID
            vec_id = generate_unique_id(action_type, idx)

            # 入库（确保metadatas键名正确）
            try:
                self.collection.add(
                    ids=[vec_id],
                    embeddings=[embedding],
                    metadatas=[{"action_type": action_type, "video_path": video_path}],
                    documents=[f"{action_type}行为视频向量"]
                )
                print(f"✅ 入库成功（ID：{vec_id}）")
                success_count += 1
            except Exception as e:
                print(f"❌ 入库失败：{type(e).__name__} - {e}")
                traceback.print_exc()

        print(f"\n===== 入库完成：成功{success_count}/{len(video_paths)}个 =====")
        return success_count

    def query_embeddings(
            self,
            query_embedding: List[float],
            action_type: str,
            n_results: int = 1
    ) -> dict:
        """查询指定动作类型的相似向量"""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"action_type": action_type},
            include=["embeddings"]
        )

    def _get_all_ids(self) -> List[str]:
        """获取集合中所有数据的ID（用于删除所有数据）"""
        try:
            all_data = self.collection.get(include=[])  # 只查ID，不查其他字段
            return all_data["ids"] if all_data.get("ids") else []
        except Exception as e:
            print(f"⚠️ 查询所有ID失败：{e}")
            return []

    def delete_embeddings(self, action_type: Optional[str] = None):
        """
        适配新版ChromaDB的删除逻辑：
        - action_type=None：删除所有数据（通过ID批量删除）
        - action_type=mopping/non_mopping：按条件删除
        """
        # 1. 删除所有数据（适配新版ChromaDB）
        if action_type is None:
            all_ids = self._get_all_ids()
            if not all_ids:
                print("✅ 向量库已为空，无需删除")
                return
            try:
                self.collection.delete(ids=all_ids)  # 通过ID删除所有数据
                print(f"✅ 已清空所有向量数据（共删除{len(all_ids)}条）")
            except Exception as e:
                print(f"❌ 清空所有数据失败：{e}")
                traceback.print_exc()
        # 2. 删除指定类型数据（原有逻辑，兼容新版）
        else:
            if action_type not in [self.config.ACTION_MOPPING, self.config.ACTION_NON_MOPPING]:
                print(f"❌ 无效动作类型：{action_type}")
                return
            try:
                self.collection.delete(where={"action_type": action_type})
                print(f"✅ 已清空 {action_type} 类型向量数据")
            except Exception as e:
                print(f"❌ 清空{action_type}类型数据失败：{e}")
                traceback.print_exc()

    def check_db_status(self) -> int:
        """查看数据库状态（统计数据量，彻底修复numpy真值判断问题）"""
        total_count = self.collection.count()
        print(f"\n📝 向量库状态：")
        print(f"   - 总数据量：{total_count}")

        if total_count > 0:
            # 仅查询metadatas（避免embeddings引发的numpy问题）
            all_data = self.collection.get(include=["metadatas"])
            # 初始化统计变量
            mop_count = 0
            non_mop_count = 0
            invalid_meta_count = 0  # 统计无效元数据数量

            # 遍历元数据，严格判断
            for meta in all_data["metadatas"]:
                if meta is None or not isinstance(meta, dict) or "action_type" not in meta:
                    invalid_meta_count += 1
                    continue
                # 正常统计
                if meta["action_type"] == self.config.ACTION_MOPPING:
                    mop_count += 1
                elif meta["action_type"] == self.config.ACTION_NON_MOPPING:
                    non_mop_count += 1
                else:
                    invalid_meta_count += 1

            # 打印统计结果
            print(f"   - 拖地视频数量：{mop_count}")
            print(f"   - 非拖地视频数量：{non_mop_count}")
            print(f"   - 无效元数据数量（无action_type）：{invalid_meta_count}")

            # 固定显示向量维度（避免numpy问题）
            print(f"   - 向量维度：512（配置默认）")

            # 提示清理无效数据
            if invalid_meta_count > 0:
                print(f"\n⚠️ 发现 {invalid_meta_count} 条无效数据，建议执行clean_invalid_data()清理！")

        return total_count

    def clean_invalid_data(self):
        """清理向量库中无action_type的无效数据（稳定版）"""
        print("===== 开始清理无效数据（无action_type） =====")
        total_count = self.collection.count()
        if total_count == 0:
            print("✅ 向量库为空，无需清理")
            return

        # 仅查询ID和元数据（完全避开embeddings）
        all_data = self.collection.get(include=["metadatas"])
        invalid_ids = []

        # 筛选无效ID（最严格的判断 + 异常捕获）
        for idx, meta in enumerate(all_data["metadatas"]):
            try:
                if meta is None or not isinstance(meta, dict) or "action_type" not in meta:
                    invalid_ids.append(all_data["ids"][idx])
            except Exception:
                invalid_ids.append(all_data["ids"][idx])

        # 执行删除
        if len(invalid_ids) > 0:
            show_ids = invalid_ids[:5] if len(invalid_ids) > 5 else invalid_ids
            print(f"❌ 发现 {len(invalid_ids)} 条无效数据，ID示例：{show_ids}...")
            # 二次确认
            confirm = input("⚠️ 确认删除这些无效数据吗？(y/n)：")
            if confirm.lower() == "y":
                self.collection.delete(ids=invalid_ids)  # 用ID删除无效数据
                print(f"✅ 已删除 {len(invalid_ids)} 条无效数据")
            else:
                print("❌ 取消删除操作")
        else:
            print("✅ 未发现无效数据")

        # 清理后查看状态
        self.check_db_status()