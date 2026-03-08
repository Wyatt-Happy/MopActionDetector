#!/usr/bin/env python3
# add_mopping_videos.py - 专门用于入库拖地（正向）视频的可执行文件
from config import MoppingDetectionConfig
from db_manager import EmbeddingDBManager

# ===================== 【用户配置区】=====================
# 请修改为你自己的拖地视频路径列表（支持单个/多个）
MOPPING_VIDEO_PATHS = [
    "E:/PoseYOLO/video/test.mp4",
    "E:/PoseYOLO/video/test1.mp4",
    "E:/PoseYOLO/video/test2.mp4",
    "E:/PoseYOLO/video/test3.mp4",
    "E:/PoseYOLO/video/test4.mp4",
    "E:/PoseYOLO/video/test5.mp4",
]
# 可选：自定义配置（比如修改采样帧数、数据库路径）
CUSTOM_CONFIG = {
    # "SAMPLE_FRAMES": 40,
    # "DB_PATH": "E:/PoseYOLO/video_embedding_db",
    # "MOPPING_THRESHOLD": 0.8
}


# ===================== 【配置结束】=====================

def main():
    print("===== 开始入库拖地（正向）视频 =====")

    # 初始化配置
    config = MoppingDetectionConfig()
    # 应用自定义配置
    for key, value in CUSTOM_CONFIG.items():
        if hasattr(config, key):
            setattr(config, key, value)
            print(f"✅ 自定义配置：{key} = {value}")

    # 初始化数据库管理器
    db_manager = EmbeddingDBManager(config)

    # 入库拖地视频
    success_count = db_manager.add_video_embeddings(
        video_paths=MOPPING_VIDEO_PATHS,
        action_type=config.ACTION_MOPPING
    )

    # 打印最终结果
    print(f"\n===== 入库完成 ======")
    print(f"总待入库数：{len(MOPPING_VIDEO_PATHS)}")
    print(f"成功入库数：{success_count}")
    print(f"失败数：{len(MOPPING_VIDEO_PATHS) - success_count}")

    # 查看入库后数据库状态
    db_manager.check_db_status()


if __name__ == "__main__":
    main()