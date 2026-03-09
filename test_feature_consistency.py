# test_feature_consistency.py
"""
测试特征一致性，确保训练和检测使用相同的特征提取方式
"""

from core.detectors.dynamic_detector import DynamicBehaviorDetector
from core.managers.behavior_manager import BehaviorManager
from core.utils.config import MoppingDetectionConfig
import os

print("测试特征一致性...")

try:
    # 初始化配置
    config = MoppingDetectionConfig()
    
    # 初始化行为管理器
    behavior_manager = BehaviorManager(config)
    
    # 清理现有行为
    behaviors = behavior_manager.list_behaviors()
    for behavior in behaviors:
        behavior_manager.delete_behavior(behavior['behavior_id'])
    
    # 创建行为
    behavior_manager.create_behavior(
        behavior_id="test_behavior",
        display_name="测试行为"
    )
    
    # 测试视频
    test_video = "e:\\PoseYOLO\\Data\\拖地\\test\\VCG42N1284190622.mp4"
    
    if not os.path.exists(test_video):
        print(f"测试视频不存在: {test_video}")
        exit(1)
    
    # 测试 1: LSTM 训练 + LSTM 检测
    print("\n测试 1: LSTM 训练 + LSTM 检测")
    detector_lstm = DynamicBehaviorDetector(config, use_temporal=True)
    
    # 添加训练视频
    success = detector_lstm.add_training_video("test_behavior", test_video)
    print(f"添加训练视频: {'成功' if success else '失败'}")
    
    # 检测
    result_lstm = detector_lstm.detect(test_video, return_all=True)
    print(f"检测结果: {result_lstm}")
    
    # 清理行为
    behavior_manager.delete_behavior("test_behavior")
    
    # 测试 2: 传统特征训练 + 传统特征检测
    print("\n测试 2: 传统特征训练 + 传统特征检测")
    detector_traditional = DynamicBehaviorDetector(config, use_temporal=False)
    
    # 重新创建行为
    behavior_manager.create_behavior(
        behavior_id="test_behavior",
        display_name="测试行为"
    )
    
    # 添加训练视频
    success = detector_traditional.add_training_video("test_behavior", test_video)
    print(f"添加训练视频: {'成功' if success else '失败'}")
    
    # 检测
    result_traditional = detector_traditional.detect(test_video, return_all=True)
    print(f"检测结果: {result_traditional}")
    
    # 测试 3: LSTM 训练 + 传统特征检测（应该失败）
    print("\n测试 3: LSTM 训练 + 传统特征检测")
    
    # 清理行为
    behavior_manager.delete_behavior("test_behavior")
    
    # 重新创建行为
    behavior_manager.create_behavior(
        behavior_id="test_behavior",
        display_name="测试行为"
    )
    
    # 使用 LSTM 添加训练视频
    success = detector_lstm.add_training_video("test_behavior", test_video)
    print(f"使用 LSTM 添加训练视频: {'成功' if success else '失败'}")
    
    # 使用传统特征检测
    result_mismatch = detector_traditional.detect(test_video, return_all=True)
    print(f"检测结果: {result_mismatch}")
    
    print("\n🎉 测试完成！")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
