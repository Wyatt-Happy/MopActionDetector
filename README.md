# PoseYOLO - 拖地行为检测系统

基于深度学习的视频行为检测系统，通过 ResNet18 提取视频特征向量，结合 ChromaDB 向量数据库实现拖地行为的智能识别。

## 项目简介

本项目是一个视频行为识别系统，专门用于检测视频中是否包含拖地行为。系统采用向量相似度匹配的方法，通过对比待测视频与已标注视频的特征相似度来判断行为类型。

### 核心特性

- 基于深度学习的视频特征提取（ResNet18）
- 向量数据库存储与检索
- 余弦相似度匹配算法
- 支持自定义检测阈值
- 模块化设计，易于扩展

## 项目结构

```
PoseYOLO/
├── config.py              # 全局配置管理
├── feature_extractor.py   # 视频特征提取器
├── db_manager.py          # 向量数据库管理器
├── action_detector.py     # 行为检测器
├── utils.py               # 工具函数
├── main.py                # 主程序入口
├── detect_mopping.py      # 独立检测脚本
├── add_mopping_videos.py  # 添加拖地视频到数据库
├── add_non_mopping_videos.py  # 添加非拖地视频到数据库
├── clear_db.py            # 清空数据库脚本
├── requirements.txt       # 依赖包列表
├── video/                 # 测试视频目录
├── no_video/              # 非拖地视频目录
└── video_embedding_db/    # 向量数据库存储目录
```

## 技术架构

### 特征提取流程

1. **视频帧读取**：使用 OpenCV 读取视频所有帧
2. **帧采样**：等间距采样 32 帧作为代表
3. **帧预处理**：调整尺寸至 224×224，归一化处理
4. **特征提取**：使用预训练 ResNet18 提取 512 维特征向量
5. **特征聚合**：对所有帧特征取均值，得到视频级特征
6. **向量标准化**：L2 归一化，便于余弦相似度计算

### 检测算法

采用双阈值判定策略：

```
判定为拖地行为需同时满足：
1. 拖地相似度 ≥ 0.75
2. 拖地相似度 - 非拖地相似度 ≥ 0.1
```

## 安装说明

### 环境要求

- Python 3.8+
- CUDA（可选，用于 GPU 加速）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖

| 依赖包 | 用途 |
|--------|------|
| torch | 深度学习框架 |
| torchvision | 图像处理与预训练模型 |
| opencv-python | 视频读取与处理 |
| chromadb | 向量数据库 |
| numpy | 数值计算 |

## 使用方法

### 第一步：添加标注视频到数据库

```python
from config import MoppingDetectionConfig
from db_manager import EmbeddingDBManager

config = MoppingDetectionConfig()
db_manager = EmbeddingDBManager(config)

# 添加拖地视频
mop_videos = ["video/mop_1.mp4", "video/mop_2.mp4"]
db_manager.add_video_embeddings(mop_videos, config.ACTION_MOPPING)

# 添加非拖地视频
non_mop_videos = ["no_video/non_mop_1.mp4", "no_video/non_mop_2.mp4"]
db_manager.add_video_embeddings(non_mop_videos, config.ACTION_NON_MOPPING)
```

### 第二步：检测视频

```python
from config import MoppingDetectionConfig
from action_detector import MoppingActionDetector

config = MoppingDetectionConfig()
detector = MoppingActionDetector(config)

# 执行检测
is_mopping, mop_sim, non_mop_sim = detector.detect("test_video.mp4")

print(f"是否拖地: {is_mopping}")
print(f"拖地相似度: {mop_sim:.3f}")
print(f"非拖地相似度: {non_mop_sim:.3f}")
```

### 快速检测脚本

直接运行独立检测脚本：

```bash
python detect_mopping.py
```

> 使用前需修改脚本中的 `TEST_VIDEO_PATH` 变量

## 配置说明

在 `config.py` 中可调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| DEVICE | auto | 计算设备（cuda/cpu） |
| FRAME_SIZE | (224, 224) | 帧缩放尺寸 |
| SAMPLE_FRAMES | 32 | 采样帧数 |
| DB_PATH | video_embedding_db | 数据库存储路径 |
| MOPPING_THRESHOLD | 0.75 | 拖地相似度阈值 |
| SIMILARITY_GAP | 0.1 | 相似度差值阈值 |
| TOP_K | 1 | 匹配Top-K样本 |

## 核心模块说明

### VideoFeatureExtractor（特征提取器）

负责将视频转换为 512 维标准化特征向量：

- `read_frames()`: 读取视频所有帧
- `sample_frames()`: 等间距采样
- `extract_frame_embedding()`: 提取单帧特征
- `extract_video_embedding()`: 提取视频级特征

### EmbeddingDBManager（数据库管理器）

管理向量数据库的增删查操作：

- `add_video_embeddings()`: 批量添加视频向量
- `query_embeddings()`: 查询相似向量
- `delete_embeddings()`: 删除指定类型数据
- `check_db_status()`: 查看数据库状态

### MoppingActionDetector（行为检测器）

核心检测逻辑实现：

- `calculate_similarity()`: 计算余弦相似度
- `detect()`: 执行行为检测

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

Copyright (c) 2026 张允泽
