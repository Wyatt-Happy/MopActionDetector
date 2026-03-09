# test_feature_space.py
"""
测试特征空间问题
"""

from core.detectors.dynamic_detector import DynamicBehaviorDetector
from core.managers.behavior_manager import BehaviorManager
from core.utils.config import MoppingDetectionConfig
import os
import numpy as np

print("测试特征空间问题...")

try:
    # 初始化配置
    config = MoppingDetectionConfig()
    
    # 初始化行为管理器
    behavior_manager = BehaviorManager(config)
    
    # 清理现有行为
    behaviors = behavior_manager.list_behaviors()
    for behavior in behaviors:
        behavior_manager.delete_behavior(behavior['behavior_id'])
        print(f"删除行为: {behavior['behavior_id']}")
    
    # 测试 1: 使用 LSTM 特征
    print("\n测试 1: 使用 LSTM 特征")
    detector_lstm = DynamicBehaviorDetector(config, use_temporal=True)
    
    # 创建行为
    behavior_manager.create_behavior(
        behavior_id="test_lstm",
        display_name="测试 LSTM",
        category="general"
    )
    
    # 测试视频
    test_video = "e:\\PoseYOLO\\Data\\拖地\\test\\VCG42N1284190622.mp4"
    
    if os.path.exists(test_video):
        # 添加训练视频
        success = detector_lstm.add_training_video("test_lstm", test_video)
        print(f"添加训练视频: {'成功' if success else '失败'}")
        
        # 提取特征
        frame_features = detector_lstm.extractor.extract_features(test_video, return_frames=True)
        temporal_feature = detector_lstm.temporal_extractor.extract(frame_features)
        
        # 计算相似度
        similarity = detector_lstm._query_behavior_similarity("test_lstm", temporal_feature)
        print(f"LSTM 特征相似度: {similarity:.3f}")
    
    # 测试 2: 使用传统特征
    print("\n测试 2: 使用传统特征")
    detector_traditional = DynamicBehaviorDetector(config, use_temporal=False)
    
    # 创建行为
    behavior_manager.create_behavior(
        behavior_id="test_traditional",
        display_name="测试传统",
        category="general"
    )
    
    if os.path.exists(test_video):
        # 添加训练视频
        success = detector_traditional.add_training_video("test_traditional", test_video)
        print(f"添加训练视频: {'成功' if success else '失败'}")
        
        # 提取特征
        traditional_feature = detector_traditional.extractor.extract_features(test_video)
        
        # 计算相似度
        similarity = detector_traditional._query_behavior_similarity("test_traditional", traditional_feature)
        print(f"传统特征相似度: {similarity:.3f}")
    
    # 测试 3: 特征交叉测试
    print("\n测试 3: 特征交叉测试")
    if os.path.exists(test_video):
        # LSTM 特征 vs 传统数据库
        similarity = detector_lstm._query_behavior_similarity("test_traditional", temporal_feature)
        print(f"LSTM 特征 vs 传统数据库: {similarity:.3f}")
        
        # 传统特征 vs LSTM 数据库
        similarity = detector_traditional._query_behavior_similarity("test_lstm", traditional_feature)
        print(f"传统特征 vs LSTM 数据库: {similarity:.3f}")
    
    print("\n🎉 测试完成！")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
