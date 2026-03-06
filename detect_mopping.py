#!/usr/bin/env python3
# detect_mopping.py - 专门用于检测视频是否包含拖地行为的可执行文件
from config import MoppingDetectionConfig
from action_detector import MoppingActionDetector

# ===================== 【用户配置区】=====================
# 请修改为你要检测的视频路径
TEST_VIDEO_PATH = "E:/PoseYOLO/test_video.mp4"

# 可选：自定义检测阈值
CUSTOM_THRESHOLD = 0.75  # 拖地相似度阈值
CUSTOM_SIMILARITY_GAP = 0.1  # 正向-反向相似度差值阈值

# 数据库路径（和config里保持一致）
DB_PATH = "E:/PoseYOLO/video_embedding_db"


# ===================== 【配置结束】=====================

def main():
    print("===== 开始检测视频是否包含拖地行为 =====")

    # 初始化配置
    config = MoppingDetectionConfig()
    config.DB_PATH = DB_PATH  # 应用自定义数据库路径

    # 初始化检测器
    detector = MoppingActionDetector(config)

    # 执行检测
    is_mopping, mop_sim, non_mop_sim = detector.detect(
        video_path=TEST_VIDEO_PATH,
        threshold=CUSTOM_THRESHOLD,
        similarity_gap=CUSTOM_SIMILARITY_GAP
    )

    # 输出最终结论
    print(f"\n===== 检测完成 ======")
    print(f"检测视频：{TEST_VIDEO_PATH}")
    print(f"最终结论：{'✅ 该视频包含拖地行为' if is_mopping else '❌ 该视频不包含拖地行为'}")


if __name__ == "__main__":
    main()