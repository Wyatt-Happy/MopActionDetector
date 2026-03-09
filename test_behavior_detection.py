# test_behavior_detection.py
"""
测试行为检测，找出 LSTM 置信度为 0% 的问题
"""

from core.detectors.dynamic_detector import DynamicBehaviorDetector
from core.managers.behavior_manager import BehaviorManager
from core.utils.config import MoppingDetectionConfig
import os

print("测试行为检测...")

try:
    # 初始化配置
    config = MoppingDetectionConfig()
    
    # 初始化行为管理器
    behavior_manager = BehaviorManager(config)
    
    # 初始化检测器（启用 LSTM）
    detector_with_lstm = DynamicBehaviorDetector(config, use_temporal=True)
    detector_without_lstm = DynamicBehaviorDetector(config, use_temporal=False)
    
    # 检查现有行为
    behaviors = behavior_manager.list_behaviors()
    print(f"现有行为: {[b['behavior_id'] for b in behaviors]}")
    
    # 清理现有行为（重新测试）
    for behavior in behaviors:
        behavior_manager.delete_behavior(behavior['behavior_id'])
        print(f"删除行为: {behavior['behavior_id']}")
    
    # 重新创建行为
    print("\n创建行为...")
    behavior_manager.create_behavior(
        behavior_id="mopping",
        display_name="拖地",
        category="hygiene"
    )
    
    behavior_manager.create_behavior(
        behavior_id="phone",
        display_name="玩手机",
        category="violation"
    )
    
    # 添加训练视频
    print("\n添加训练视频...")
    
    # 拖地训练视频
    mopping_videos = [
        "e:\\PoseYOLO\\Data\\拖地\\train\\test.mp4",
        "e:\\PoseYOLO\\Data\\拖地\\train\\test1.mp4",
        "e:\\PoseYOLO\\Data\\拖地\\train\\test2.mp4"
    ]
    
    for video in mopping_videos:
        if os.path.exists(video):
            success = detector_with_lstm.add_training_video("mopping", video)
            print(f"添加拖地视频 {os.path.basename(video)}: {'成功' if success else '失败'}")
        else:
            print(f"视频不存在: {video}")
    
    # 玩手机训练视频
    phone_videos = [
        "e:\\PoseYOLO\\Data\\玩手机\\train\\玩手机1.mp4",
        "e:\\PoseYOLO\\Data\\玩手机\\train\\玩手机2.mp4",
        "e:\\PoseYOLO\\Data\\玩手机\\train\\玩手机3.mp4"
    ]
    
    for video in phone_videos:
        if os.path.exists(video):
            success = detector_with_lstm.add_training_video("phone", video)
            print(f"添加玩手机视频 {os.path.basename(video)}: {'成功' if success else '失败'}")
        else:
            print(f"视频不存在: {video}")
    
    # 测试视频
    test_video = "e:\\PoseYOLO\\Data\\拖地\\test\\VCG42N1284190622.mp4"
    
    if os.path.exists(test_video):
        print(f"\n测试视频: {os.path.basename(test_video)}")
        
        # 使用 LSTM 检测
        print("\n使用 LSTM 检测:")
        result_lstm = detector_with_lstm.detect(test_video, return_all=True)
        print(f"  结果: {result_lstm}")
        
        # 不使用 LSTM 检测
        print("\n不使用 LSTM 检测:")
        result_no_lstm = detector_without_lstm.detect(test_video, return_all=True)
        print(f"  结果: {result_no_lstm}")
    else:
        print(f"测试视频不存在: {test_video}")
    
    # 检查数据库状态
    print("\n数据库状态:")
    stats = detector_with_lstm.get_behavior_stats()
    print(f"  统计: {stats}")
    
    print("\n🎉 测试完成！")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
