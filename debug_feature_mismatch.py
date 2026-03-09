# debug_feature_mismatch.py
"""
调试特征不匹配问题
"""

from core.detectors.dynamic_detector import DynamicBehaviorDetector
from core.managers.behavior_manager import BehaviorManager
from core.utils.config import MoppingDetectionConfig
import os
import numpy as np

print("调试特征不匹配问题...")

try:
    # 初始化配置
    config = MoppingDetectionConfig()
    
    # 初始化行为管理器
    behavior_manager = BehaviorManager(config)
    
    # 初始化检测器
    detector = DynamicBehaviorDetector(config, use_temporal=True)
    
    # 检查现有行为
    behaviors = behavior_manager.list_behaviors()
    print(f"现有行为: {[b['behavior_id'] for b in behaviors]}")
    
    for behavior in behaviors:
        behavior_id = behavior['behavior_id']
        print(f"\n检查行为: {behavior_id}")
        
        # 检查行为元数据
        print(f"  行为元数据: {behavior}")
        
        # 检查特征类型
        feature_type = behavior.get('feature_type', 'unknown')
        print(f"  特征类型: {feature_type}")
        
        # 检查数据库中的特征
        try:
            collection = behavior_manager.get_collection(behavior_id)
            count = collection.count()
            print(f"  数据库中视频数量: {count}")
            
            # 查看第一个特征的维度
            if count > 0:
                results = collection.get(limit=1)
                if results and results['embeddings']:
                    embedding = results['embeddings'][0]
                    print(f"  特征维度: {len(embedding)}")
                    print(f"  特征示例: {embedding[:5]}")
        except Exception as e:
            print(f"  检查数据库失败: {e}")
    
    # 测试特征提取
    test_video = "e:\\PoseYOLO\\Data\\拖地\\test\\VCG42N1284190622.mp4"
    if os.path.exists(test_video):
        print(f"\n测试视频: {test_video}")
        
        # 提取帧级特征
        frame_features = detector.extractor.extract_features(test_video, return_frames=True)
        print(f"  帧级特征数量: {len(frame_features)}")
        print(f"  帧级特征维度: {len(frame_features[0])}")
        
        # 提取时序特征
        temporal_feature = detector.temporal_extractor.extract(frame_features)
        print(f"  时序特征维度: {len(temporal_feature)}")
        print(f"  时序特征示例: {temporal_feature[:5]}")
        
        # 计算与数据库中特征的相似度
        for behavior in behaviors:
            behavior_id = behavior['behavior_id']
            print(f"\n与 {behavior_id} 的相似度:")
            
            try:
                collection = behavior_manager.get_collection(behavior_id)
                results = collection.query(
                    query_embeddings=[temporal_feature.tolist()],
                    n_results=1
                )
                
                if results and results['distances'] and results['distances'][0]:
                    distance = results['distances'][0][0]
                    similarity = 1 - distance
                    print(f"  相似度: {similarity:.3f}")
                else:
                    print(f"  无相似结果")
                    
            except Exception as e:
                print(f"  计算相似度失败: {e}")
    
    print("\n🎉 调试完成！")
    
except Exception as e:
    print(f"❌ 调试失败: {e}")
    import traceback
    traceback.print_exc()
