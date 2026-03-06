# main.py
from mopping_detection import (
    MoppingDetectionConfig,
    EmbeddingDBManager,
    MoppingActionDetector
)

# ===================== 示例：使用流程 =====================
if __name__ == "__main__":
    # 1. 初始化配置（可自定义修改参数）
    config = MoppingDetectionConfig()
    # 可选：修改配置（比如调整阈值、采样帧数）
    # config.MOPPING_THRESHOLD = 0.8
    # config.SAMPLE_FRAMES = 40

    # 2. 初始化数据库管理器
    db_manager = EmbeddingDBManager(config)

    # 3. 【第一步】添加标注视频到数据库（首次使用必须执行）
    # 拖地视频路径列表
    mop_video_paths = [
        "path/to/mop_video_1.mp4",
        "path/to/mop_video_2.mp4"
    ]
    # 非拖地视频路径列表
    non_mop_video_paths = [
        "path/to/non_mop_video_1.mp4",
        "path/to/non_mop_video_2.mp4"
    ]
    # 入库（注释掉表示已入库，无需重复执行）
    # db_manager.add_video_embeddings(mop_video_paths, config.ACTION_MOPPING)
    # db_manager.add_video_embeddings(non_mop_video_paths, config.ACTION_NON_MOPPING)

    # 4. 查看数据库状态
    db_manager.check_db_status()

    # 5. 【第二步】检测待测视频
    detector = MoppingActionDetector(config)
    test_video_path = "path/to/test_video.mp4"
    is_mopping, mop_sim, non_mop_sim = detector.detect(test_video_path)

    # 6. 可选：清空数据库（测试用）
    # db_manager.delete_embeddings(config.ACTION_MOPPING)  # 仅清空拖地数据
    # db_manager.delete_embeddings()  # 清空所有数据