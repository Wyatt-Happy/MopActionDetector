# PoseYOLO - 员工行为检测系统 v3.0

![UI界面](./UI.png)

基于深度学习的动态视频行为检测系统，支持用户自定义行为类型，通过 ResNet18/50 提取视频特征向量，结合 ChromaDB 向量数据库实现智能行为识别。

## 项目简介

本项目是一个**动态行为检测系统**，用户可以通过 Web 界面自定义员工行为类型（如：拖地、收银、玩手机等），上传示例视频进行训练，系统会自动学习并识别这些行为。

### 核心特性

- **动态行为定义** - 用户可自定义任意行为类型
- **多行为检测** - 同时支持多种行为的识别
- **批量视频导入** - 支持批量上传训练视频
- **基于深度学习** - ResNet18/ResNet50 特征提取
- **时序建模** - LSTM 捕获动作序列信息
- **向量数据库** - ChromaDB 存储与检索
- **INT8 量化** - 模型加速优化
- **实时流检测** - 支持摄像头实时检测
- **多线程并行** - 批量视频处理
- **RESTful API** - FastAPI 服务接口
- **Web 可视化** - 直观的操作界面
- **Swagger 文档** - 在线 API 测试

## 项目结构

```
PoseYOLO/
├── api.py                       # FastAPI 服务入口
├── config.yaml                  # YAML 配置文件
├── requirements.txt             # 依赖包列表
│
├── core/                        # 核心模块
│   ├── utils/
│   │   └── config.py            # 配置管理类
│   ├── extractors/
│   │   ├── feature_extractor.py # 视频特征提取器（旧版）
│   │   └── universal_extractor.py # 通用特征提取器（新版）
│   ├── detectors/
│   │   ├── action_detector.py   # 行为检测器（旧版）
│   │   ├── dynamic_detector.py  # 动态行为检测器（新版）
│   │   └── stream_detector.py   # 实时流检测器
│   └── managers/
│       ├── db_manager.py        # 向量数据库管理器
│       ├── behavior_manager.py  # 动态行为管理器
│       ├── export_manager.py    # 结果导出管理器
│       └── logger.py            # 日志管理器
│
├── scripts/                     # 脚本工具
│   ├── main.py                  # 主程序入口
│   ├── detect_mopping.py        # 拖地检测脚本（示例）
│   ├── add_mopping_videos.py    # 添加拖地视频
│   ├── add_non_mopping_videos.py # 添加非拖地视频
│   └── clear_db.py              # 清空数据库
│
├── configs/                     # 配置文件目录
│   └── config.yaml
│
├── static/                      # Web 静态文件
│   └── index.html
│
├── video_embedding_db/          # 向量数据库存储
├── behavior_db/                 # 动态行为数据库
├── logs/                        # 日志目录
└── exports/                     # 导出结果目录
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
| fastapi | Web API 框架 |
| uvicorn | ASGI 服务器 |
| pyyaml | YAML 配置解析 |
| numpy | 数值计算 |

## 快速开始

### 1. 启动服务

```bash
python api.py
```

服务启动后访问：
- Web 界面：http://localhost:8000/ui
- API 文档：http://localhost:8000/docs
- ReDoc 文档：http://localhost:8000/redoc

### 2. 使用 Web 界面

#### 创建行为类型
1. 点击「➕ 创建行为」标签页
2. 输入行为ID（如：`cashier`）
3. 输入显示名称（如：`收银服务`）
4. 选择分类（卫生/服务/安全/违规/一般）
5. 点击「创建行为」

#### 添加训练视频
1. 点击「🎓 训练数据」标签页
2. 选择行为类型
3. 上传该行为的示例视频（支持多选）
4. 系统自动提取特征并存储

#### 检测视频
1. 点击「🔍 行为检测」标签页
2. 上传待检测视频
3. 系统自动与所有行为比对
4. 显示最匹配的行为及置信度

#### 实时摄像头检测
1. 点击「📹 实时检测」标签页
2. 点击「启动摄像头」按钮
3. 系统自动从摄像头捕获视频并检测
4. 实时显示检测结果

## API 接口

### 动态行为管理

#### 创建行为
```bash
curl -X POST "http://localhost:8000/behaviors/create" \
  -F "behavior_id=cashier" \
  -F "display_name=收银服务" \
  -F "category=service" \
  -F "description=员工在收银台进行收银操作"
```

#### 列出所有行为
```bash
curl "http://localhost:8000/behaviors/list"
```

#### 删除行为
```bash
curl -X DELETE "http://localhost:8000/behaviors/cashier"
```

#### 添加单个训练视频
```bash
curl -X POST "http://localhost:8000/behaviors/cashier/add_video" \
  -F "file=@cashier_demo.mp4"
```

#### 批量添加训练视频
```bash
curl -X POST "http://localhost:8000/behaviors/cashier/add_videos_batch" \
  -F "files=@video1.mp4" \
  -F "files=@video2.mp4" \
  -F "files=@video3.mp4"
```

返回示例：
```json
{
  "success": true,
  "behavior_id": "cashier",
  "total": 3,
  "success_count": 3,
  "failed_count": 0,
  "video_count": 15
}
```

### 行为检测

#### 动态检测
```bash
curl -X POST "http://localhost:8000/detect/dynamic" \
  -F "file=@test_video.mp4" \
  -F "return_all=true"
```

返回示例：
```json
{
  "behavior_id": "cashier",
  "behavior_name": "收银服务",
  "confidence": 0.92,
  "is_confident": true,
  "conclusion": "该视频最可能是：收银服务（置信度92%）",
  "all_results": [
    {"behavior_id": "cashier", "similarity": 0.92},
    {"behavior_id": "mopping", "similarity": 0.15}
  ]
}
```

### 实时流检测

#### 启动摄像头
```bash
curl -X POST "http://localhost:8000/stream/start" \
  -H "Content-Type: application/json" \
  -d '{"source": "0"}'
```

#### 获取检测结果
```bash
curl "http://localhost:8000/stream/result"
```

#### 停止检测
```bash
curl -X POST "http://localhost:8000/stream/stop"
```

## 行为分类

系统支持以下行为分类：

| 分类 | 说明 | 示例 |
|------|------|------|
| `hygiene` | 卫生行为 | 拖地、擦桌子、洗手 |
| `service` | 服务行为 | 收银、接待、导购 |
| `safety` | 安全行为 | 检查设备、穿戴防护 |
| `violation` | 违规行为 | 玩手机、吸烟、打架 |
| `general` | 一般行为 | 其他行为 |

## 技术架构

### 工作流程

```
1. 创建行为类型
   ↓
2. 上传示例视频（训练数据）
   ↓
3. 系统自动提取特征并存储到 ChromaDB
   ↓
4. 上传待检测视频
   ↓
5. 系统与所有行为进行相似度比对
   ↓
6. 返回最匹配的行为及置信度
```

### 特征提取流程

1. **视频帧读取**：使用 OpenCV 读取视频
2. **帧采样**：等间距采样 32 帧作为代表
3. **帧预处理**：调整尺寸至 224×224，归一化处理
4. **特征提取**：使用预训练 ResNet18/50 提取特征向量
5. **特征聚合**：对所有帧特征取均值
6. **向量标准化**：L2 归一化

### 相似度计算

使用余弦相似度进行特征比对：

```
similarity = 1 - distance(query_feature, stored_feature)
```

## 配置文件

`config.yaml` 主要配置项：

```yaml
# 设备配置
device:
  type: "auto"  # cuda / cpu / auto

# 模型配置
model:
  backbone: "resnet18"  # resnet18 / resnet50
  quantization:
    enabled: false      # INT8 量化

# API 配置
api:
  host: "0.0.0.0"
  port: 8000
  web_ui_enabled: true
```

## 性能优化

### INT8 量化

```yaml
model:
  quantization:
    enabled: true
    backend: "fbgemm"
```

### 并行处理

```yaml
parallel:
  enabled: true
  workers: 0  # 0 = CPU 核心数
```

## 日志系统

```yaml
logging:
  level: "INFO"
  file: "logs/detection.log"
  console: true
```

## API 文档

启动服务后，可通过以下地址访问 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Swagger UI 提供在线测试功能，可以直接在浏览器中测试所有 API 接口。

## 兼容性说明

系统保留了旧版拖地检测功能，可通过以下方式使用：

```python
# 旧版 API（兼容）
POST /detect          # 单视频拖地检测
POST /detect/batch    # 批量拖地检测
GET  /db/status       # 数据库状态
POST /db/add          # 添加视频到数据库

# 新版 API（动态行为）
POST /behaviors/create              # 创建行为
GET  /behaviors/list                # 列出行为
DELETE /behaviors/{id}              # 删除行为
POST /behaviors/{id}/add_video      # 添加单个训练视频
POST /behaviors/{id}/add_videos_batch # 批量添加训练视频
GET  /behaviors/stats               # 获取统计信息
POST /detect/dynamic                # 动态行为检测
```

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

Copyright (c) 2026 张允泽
