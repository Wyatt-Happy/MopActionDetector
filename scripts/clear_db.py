#!/usr/bin/env python3
# clear_db.py - 专门用于清除向量库数据的可执行文件
from config import MoppingDetectionConfig
from db_manager import EmbeddingDBManager

# ===================== 【用户配置区】=====================
# 选择要清除的类型：
# - None：清除所有数据
# - "mopping"：仅清除拖地（正向）数据
# - "non_mopping"：仅清除非拖地（负向）数据
CLEAR_TYPE = None  # 可选值：None / "mopping" / "non_mopping"
# 是否先清理无效数据（建议设为True）
CLEAN_INVALID_FIRST = True
# 数据库路径（和config里保持一致）
DB_PATH = "E:/PoseYOLO/video_embedding_db"


# ===================== 【配置结束】=====================

def main():
    print("===== 开始清除向量库数据 =====")

    # 初始化配置
    config = MoppingDetectionConfig()
    config.DB_PATH = DB_PATH  # 应用自定义数据库路径

    # 初始化数据库管理器
    db_manager = EmbeddingDBManager(config)

    # 先清理无效数据（可选）
    if CLEAN_INVALID_FIRST:
        db_manager.clean_invalid_data()

    # 清除前先查看当前状态
    print("\n【清除前数据库状态】")
    total_count = db_manager.check_db_status()

    if total_count == 0:
        print("\n✅ 向量库已为空，无需清除")
        return

    # 二次确认（防止误操作）
    confirm = input(f"\n⚠️ 确认要清除【{CLEAR_TYPE or '所有'}】数据吗？(y/n)：")
    if confirm.lower() != "y":
        print("❌ 取消清除操作")
        return

    # 执行清除
    db_manager.delete_embeddings(action_type=CLEAR_TYPE)

    # 清除后查看状态
    print("\n【清除后数据库状态】")
    db_manager.check_db_status()

    print("\n===== 清除操作完成 =====")


if __name__ == "__main__":
    main()