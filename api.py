# api.py
import os
import tempfile
import shutil
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from core.utils.config import MoppingDetectionConfig
from core.detectors.action_detector import MoppingActionDetector
from core.managers.db_manager import EmbeddingDBManager
from core.detectors.stream_detector import VideoStreamDetector, ParallelVideoProcessor
from core.managers.export_manager import ExportManager, DetectionLogger
from core.managers.logger import get_logger
from core.managers.behavior_manager import BehaviorManager, BEHAVIOR_CATEGORIES
from core.detectors.dynamic_detector import DynamicBehaviorDetector

logger = get_logger(__name__)

# 初始化配置
config = MoppingDetectionConfig()

# 创建 FastAPI 应用
app = FastAPI(
    title=config.API_WEB_UI_TITLE,
    description="基于深度学习的员工行为检测 API - 支持动态行为定义",
    version="3.0.0",
    swagger_ui_parameters={
        "urls": None,
    }
)

# 使用国内 CDN 加载 Swagger UI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import Request
from fastapi.responses import HTMLResponse


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_js_url="https://cdn.bootcdn.net/ajax/libs/swagger-ui/4.10.3/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.bootcdn.net/ajax/libs/swagger-ui/4.10.3/swagger-ui.css",
    )

# 配置 CORS
if config.API_CORS_ENABLED:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.API_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 全局组件
detector = MoppingActionDetector(config)
db_manager = EmbeddingDBManager(config)
export_manager = ExportManager(config)

# 动态行为检测组件（新增）
behavior_manager = BehaviorManager(config)
dynamic_detector = DynamicBehaviorDetector(config)

# 临时文件目录
TEMP_DIR = tempfile.mkdtemp()


# ==================== 数据模型 ====================

class DetectionResult(BaseModel):
    is_mopping: bool
    mop_similarity: float
    non_mop_similarity: float
    details: dict


class BatchDetectionRequest(BaseModel):
    video_paths: List[str]


class BatchDetectionResponse(BaseModel):
    results: List[dict]
    total_count: int
    mopping_count: int


class DBStatusResponse(BaseModel):
    total_count: int
    mopping_count: int
    non_mopping_count: int


class StreamStartRequest(BaseModel):
    source: str = "0"  # 摄像头索引或 RTMP/RTSP URL


# ==================== API 路由 ====================

@app.get("/")
async def root():
    """API 根路径"""
    return {
        "message": "拖地行为检测 API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "device": config.DEVICE,
        "model": config.MODEL_BACKBONE
    }


# -------------------- 单视频检测 --------------------

@app.post("/detect", response_model=DetectionResult)
async def detect_video(
    file: UploadFile = File(...),
    threshold: float = Form(None),
    similarity_gap: float = Form(None)
):
    """
    上传单个视频进行检测
    
    - **file**: 视频文件 (mp4/avi/mov/mkv)
    - **threshold**: 可选，自定义拖地相似度阈值
    - **similarity_gap**: 可选，自定义相似度差值阈值
    """
    # 验证文件类型
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in config.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的文件格式: {ext}"
        )
    
    # 保存上传的文件
    temp_path = os.path.join(TEMP_DIR, f"upload_{datetime.now().timestamp()}{ext}")
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"接收到上传文件: {file.filename}")
        
        # 执行检测
        is_mopping, mop_sim, non_mop_sim, details = detector.detect(
            temp_path,
            threshold=threshold,
            similarity_gap=similarity_gap
        )
        
        return DetectionResult(
            is_mopping=is_mopping,
            mop_similarity=mop_sim,
            non_mop_similarity=non_mop_sim,
            details=details
        )
        
    except Exception as e:
        logger.error(f"检测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)


# -------------------- 批量检测 --------------------

@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(files: List[UploadFile] = File(...)):
    """
    批量上传视频进行检测
    
    - **files**: 多个视频文件
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="一次最多上传 10 个文件")
    
    temp_paths = []
    try:
        # 保存所有上传的文件
        for file in files:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in config.SUPPORTED_FORMATS:
                continue
                
            temp_path = os.path.join(TEMP_DIR, f"batch_{datetime.now().timestamp()}_{file.filename}")
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            temp_paths.append((temp_path, file.filename))
        
        # 使用并行处理器
        processor = ParallelVideoProcessor(config)
        video_paths = [p[0] for p in temp_paths]
        results = processor.submit(video_paths)
        
        # 添加原始文件名
        for i, (temp_path, original_name) in enumerate(temp_paths):
            results[i]['original_filename'] = original_name
        
        # 统计
        mopping_count = sum(1 for r in results if r['is_mopping'])
        
        return BatchDetectionResponse(
            results=results,
            total_count=len(results),
            mopping_count=mopping_count
        )
        
    except Exception as e:
        logger.error(f"批量检测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        for temp_path, _ in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# -------------------- 数据库管理 --------------------

@app.get("/db/status", response_model=DBStatusResponse)
async def get_db_status():
    """获取数据库状态"""
    try:
        total = db_manager.check_db_status()
        # 获取分类统计
        all_data = db_manager.collection.get(include=["metadatas"])
        mop_count = sum(1 for m in all_data["metadatas"] 
                       if m.get("action_type") == config.ACTION_MOPPING)
        non_mop_count = sum(1 for m in all_data["metadatas"]
                           if m.get("action_type") == config.ACTION_NON_MOPPING)
        
        return DBStatusResponse(
            total_count=total,
            mopping_count=mop_count,
            non_mopping_count=non_mop_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/db/add")
async def add_video_to_db(
    file: UploadFile = File(...),
    action_type: str = Form(...)
):
    """
    添加视频到数据库
    
    - **file**: 视频文件
    - **action_type**: 动作类型 (mopping / non_mopping)
    """
    if action_type not in [config.ACTION_MOPPING, config.ACTION_NON_MOPPING]:
        raise HTTPException(
            status_code=400,
            detail=f"无效的动作类型: {action_type}"
        )
    
    ext = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(TEMP_DIR, f"db_add_{datetime.now().timestamp()}{ext}")
    
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        count = db_manager.add_video_embeddings([temp_path], action_type)
        
        return {
            "success": count > 0,
            "added_count": count,
            "action_type": action_type
        }
        
    except Exception as e:
        logger.error(f"添加视频失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.delete("/db/clear")
async def clear_db(action_type: str = None):
    """
    清空数据库
    
    - **action_type**: 可选，指定清空类型 (mopping / non_mopping)，不传则清空全部
    """
    try:
        db_manager.delete_embeddings(action_type)
        return {
            "success": True,
            "message": f"已清空 {action_type or '全部'} 数据"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- 结果导出 --------------------

@app.post("/export")
async def export_results(
    background_tasks: BackgroundTasks,
    format: str = "csv"
):
    """
    导出检测结果
    
    - **format**: 导出格式 (csv / json / both)
    """
    # 这里需要从某处获取结果，简化处理
    # 实际应用中可能需要先执行检测再导出
    return {
        "message": "请使用 /detect 或 /detect/batch 后导出结果",
        "export_format": format
    }


# -------------------- 实时流检测 --------------------

stream_detector = None

@app.post("/stream/start")
async def start_stream(request: StreamStartRequest):
    """
    启动实时流检测
    
    - **source**: 视频源 (0 为默认摄像头，或 RTMP/RTSP URL)
    """
    global stream_detector
    
    if stream_detector and stream_detector.is_running:
        return {"message": "流检测已在运行"}
    
    stream_detector = VideoStreamDetector(config)
    success = stream_detector.start_capture(request.source)
    
    if success:
        return {
            "success": True,
            "message": "流检测已启动",
            "source": request.source
        }
    else:
        raise HTTPException(status_code=500, detail="无法启动视频流")


@app.get("/stream/result")
async def get_stream_result():
    """获取实时流最新检测结果"""
    global stream_detector
    
    if not stream_detector or not stream_detector.is_running:
        return {
            "status": "not_started",
            "message": "流检测未启动"
        }
    
    result = stream_detector.get_latest_result(timeout=0.5)
    
    if result:
        return result
    else:
        return {
            "status": "running",
            "message": "暂无新结果"
        }


@app.post("/stream/stop")
async def stop_stream():
    """停止实时流检测"""
    global stream_detector
    
    if stream_detector:
        stream_detector.stop()
        stream_detector = None
    
    return {"success": True, "message": "流检测已停止"}


# -------------------- 动态行为管理 API --------------------

@app.post("/behaviors/create")
async def create_behavior(
    behavior_id: str = Form(...),
    display_name: str = Form(...),
    category: str = Form("general"),
    description: str = Form(""),
    color: str = Form("#667eea")
):
    """
    创建新行为类型
    
    - **behavior_id**: 行为唯一标识（如：mopping, cashier）
    - **display_name**: 显示名称（如：拖地, 收银）
    - **category**: 分类（hygiene/service/safety/violation/general）
    - **description**: 行为描述
    - **color**: UI显示颜色
    """
    try:
        behavior_info = behavior_manager.create_behavior(
            behavior_id=behavior_id,
            display_name=display_name,
            category=category,
            description=description,
            color=color
        )
        return {
            "success": True,
            "behavior": behavior_info
        }
    except Exception as e:
        logger.error(f"创建行为失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/behaviors/list")
async def list_behaviors(category: str = None):
    """
    列出所有行为类型
    
    - **category**: 可选，按分类筛选
    """
    try:
        behaviors = behavior_manager.list_behaviors(category)
        return {
            "behaviors": behaviors,
            "total": len(behaviors)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/behaviors/categories")
async def get_behavior_categories():
    """获取行为分类列表"""
    return BEHAVIOR_CATEGORIES


@app.delete("/behaviors/{behavior_id}")
async def delete_behavior(behavior_id: str):
    """删除行为类型"""
    try:
        success = behavior_manager.delete_behavior(behavior_id)
        return {
            "success": success,
            "message": "删除成功" if success else "删除失败"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/behaviors/{behavior_id}/add_video")
async def add_video_to_behavior(
    behavior_id: str,
    file: UploadFile = File(...),
    metadata: str = Form("")
):
    """
    为指定行为添加训练视频
    
    - **behavior_id**: 行为ID
    - **file**: 视频文件
    - **metadata**: 可选的JSON格式元数据
    """
    try:
        # 检查行为是否存在
        behavior = behavior_manager.get_behavior(behavior_id)
        if not behavior:
            raise HTTPException(status_code=404, detail=f"行为 {behavior_id} 不存在")
        
        # 保存临时文件
        ext = os.path.splitext(file.filename)[1].lower()
        temp_path = os.path.join(TEMP_DIR, f"behavior_{behavior_id}_{datetime.now().timestamp()}{ext}")
        
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # 添加到行为数据库
        import json
        meta = json.loads(metadata) if metadata else {}
        success = dynamic_detector.add_training_video(behavior_id, temp_path, meta)
        
        return {
            "success": success,
            "behavior_id": behavior_id,
            "video_count": behavior_manager._get_behavior_video_count(behavior_id)
        }
        
    except Exception as e:
        logger.error(f"添加训练视频失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/behaviors/stats")
async def get_behavior_stats():
    """获取行为统计信息"""
    try:
        stats = dynamic_detector.get_behavior_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/behaviors/{behavior_id}/add_videos_batch")
async def add_videos_batch(
    behavior_id: str,
    files: List[UploadFile] = File(...)
):
    """
    批量为指定行为添加训练视频
    
    - **behavior_id**: 行为ID
    - **files**: 多个视频文件
    """
    try:
        # 检查行为是否存在
        behavior = behavior_manager.get_behavior(behavior_id)
        if not behavior:
            raise HTTPException(status_code=404, detail=f"行为 {behavior_id} 不存在")
        
        results = []
        success_count = 0
        failed_count = 0
        
        for file in files:
            try:
                # 保存临时文件
                ext = os.path.splitext(file.filename)[1].lower()
                temp_path = os.path.join(TEMP_DIR, f"behavior_{behavior_id}_{datetime.now().timestamp()}_{file.filename}{ext}")
                
                with open(temp_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                
                # 添加到行为数据库
                success = dynamic_detector.add_training_video(behavior_id, temp_path, {"filename": file.filename})
                
                if success:
                    success_count += 1
                    results.append({"filename": file.filename, "success": True})
                else:
                    failed_count += 1
                    results.append({"filename": file.filename, "success": False, "error": "添加失败"})
                
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                failed_count += 1
                results.append({"filename": file.filename, "success": False, "error": str(e)})
        
        return {
            "success": True,
            "behavior_id": behavior_id,
            "total": len(files),
            "success_count": success_count,
            "failed_count": failed_count,
            "results": results,
            "video_count": behavior_manager._get_behavior_video_count(behavior_id)
        }
        
    except Exception as e:
        logger.error(f"批量添加训练视频失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- 动态行为检测 API --------------------

@app.post("/detect/dynamic")
async def detect_behavior_dynamic(
    file: UploadFile = File(...),
    return_all: bool = Form(False)
):
    """
    动态行为检测
    
    与所有已定义行为进行相似度比对，返回最匹配的行为
    
    - **file**: 视频文件
    - **return_all**: 是否返回所有行为的相似度
    """
    ext = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(TEMP_DIR, f"detect_dynamic_{datetime.now().timestamp()}{ext}")
    
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        result = dynamic_detector.detect(temp_path, return_all=return_all)
        
        return result
        
    except Exception as e:
        logger.error(f"动态检测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# -------------------- Web UI --------------------

if config.API_WEB_UI_ENABLED:
    # 创建静态文件目录
    STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR, exist_ok=True)
    
    # 挂载静态文件
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    
    @app.get("/ui", response_class=FileResponse)
    async def web_ui():
        """Web 界面"""
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
            return index_path
        else:
            # 返回简单的 HTML 界面
            return create_simple_ui()


def create_simple_ui():
    """创建增强版 Web UI HTML - 支持动态行为管理"""
    html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>员工行为检测系统 v3.0</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        .card h2 {
            color: #333;
            margin-bottom: 20px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            background: #f8f9ff;
            border-color: #764ba2;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s;
            margin-right: 10px;
            margin-bottom: 10px;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn-secondary { background: #6c757d; }
        .btn-success { background: #28a745; }
        .btn-danger { background: #dc3545; }
        .btn-warning { background: #ffc107; color: #333; }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        .result.success { background: #d4edda; color: #155724; }
        .result.error { background: #f8d7da; color: #721c24; }
        .result.info { background: #d1ecf1; color: #0c5460; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .stat-item {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label { color: #666; margin-top: 5px; }
        #fileInput, #importFileInput, #behaviorVideoInput, #detectFileInput { display: none; }
        .progress {
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 20px;
            display: none;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e9ecef;
            flex-wrap: wrap;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
        }
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .behavior-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .behavior-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid #667eea;
            transition: transform 0.2s;
        }
        .behavior-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        .behavior-card h3 {
            color: #333;
            margin-bottom: 10px;
        }
        .behavior-card .category {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 12px;
            color: white;
            margin-bottom: 10px;
        }
        .behavior-card .video-count {
            color: #666;
            font-size: 14px;
        }
        .behavior-card .actions {
            margin-top: 15px;
        }
        .behavior-card .actions button {
            padding: 6px 12px;
            font-size: 14px;
            margin-right: 5px;
        }
        .detection-result {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        .detection-result h3 {
            color: #333;
            margin-bottom: 15px;
        }
        .similarity-bar {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .similarity-bar .label {
            width: 120px;
            font-weight: 500;
        }
        .similarity-bar .bar {
            flex: 1;
            height: 25px;
            background: #e9ecef;
            border-radius: 12px;
            overflow: hidden;
            margin: 0 10px;
        }
        .similarity-bar .bar-fill {
            height: 100%;
            border-radius: 12px;
            transition: width 0.5s;
        }
        .similarity-bar .value {
            width: 60px;
            text-align: right;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>👔 员工行为检测系统 v3.0</h1>
        
        <div class="card">
            <div class="tabs">
                <div class="tab active" onclick="switchTab('detect')">🔍 行为检测</div>
                <div class="tab" onclick="switchTab('stream')">📹 实时检测</div>
                <div class="tab" onclick="switchTab('behaviors')">📋 行为管理</div>
                <div class="tab" onclick="switchTab('create')">➕ 创建行为</div>
                <div class="tab" onclick="switchTab('train')">🎓 训练数据</div>
            </div>
            
            <!-- 检测标签页 -->
            <div id="detectTab" class="tab-content active">
                <h2>上传视频进行行为检测</h2>
                <p style="color: #666; margin-bottom: 20px;">系统会自动与所有已定义的行为进行比对，返回最匹配的结果</p>
                <div class="upload-area" onclick="document.getElementById('detectFileInput').click()">
                    <p>📹 点击上传视频进行检测</p>
                    <p style="color: #999; margin-top: 10px;">支持 MP4, AVI, MOV, MKV 格式</p>
                </div>
                <input type="file" id="detectFileInput" accept=".mp4,.avi,.mov,.mkv" onchange="handleDetectFile(event)">
                <div class="progress" id="detectProgress">
                    <div class="progress-bar" id="detectProgressBar"></div>
                </div>
                <div class="result" id="detectResult"></div>
                <div id="detectionDetails"></div>
            </div>
            
            <!-- 实时检测标签页 -->
            <div id="streamTab" class="tab-content">
                <h2>实时摄像头行为检测</h2>
                <p style="color: #666; margin-bottom: 20px;">启动摄像头进行实时行为检测，系统会自动识别视频中的行为类型</p>
                <div style="margin-bottom: 20px;">
                    <button class="btn btn-success" onclick="startStream()">▶️ 启动摄像头</button>
                    <button class="btn btn-danger" onclick="stopStream()">⏹️ 停止检测</button>
                </div>
                <div id="streamStatus" class="result info" style="display: block;">
                    <p>📷 点击「启动摄像头」开始实时检测</p>
                </div>
                <div id="streamResult"></div>
            </div>
            
            <!-- 行为管理标签页 -->
            <div id="behaviorsTab" class="tab-content">
                <h2>已定义的行为类型</h2>
                <div style="margin-bottom: 20px;">
                    <button class="btn" onclick="loadBehaviors()">🔄 刷新列表</button>
                    <span id="behaviorStats" style="margin-left: 20px; color: #666;"></span>
                </div>
                <div class="behavior-list" id="behaviorList">
                    <!-- 行为列表将在这里动态生成 -->
                </div>
            </div>
            
            <!-- 创建行为标签页 -->
            <div id="createTab" class="tab-content">
                <h2>创建新行为类型</h2>
                <form id="createBehaviorForm" onsubmit="return false;">
                    <div class="form-group">
                        <label>行为ID（英文标识，如：cashier）</label>
                        <input type="text" id="behaviorId" placeholder="例如：cashier, mopping, phone_usage" required>
                    </div>
                    <div class="form-group">
                        <label>显示名称（中文）</label>
                        <input type="text" id="displayName" placeholder="例如：收银服务、拖地清洁" required>
                    </div>
                    <div class="form-group">
                        <label>分类</label>
                        <select id="category">
                            <option value="general">一般行为</option>
                            <option value="hygiene">卫生行为</option>
                            <option value="service">服务行为</option>
                            <option value="safety">安全行为</option>
                            <option value="violation">违规行为</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>描述</label>
                        <textarea id="description" rows="3" placeholder="描述该行为的特征..."></textarea>
                    </div>
                    <button type="button" class="btn btn-success" onclick="createBehavior()">✅ 创建行为</button>
                </form>
                <div class="result" id="createResult"></div>
            </div>
            
            <!-- 训练数据标签页 -->
            <div id="trainTab" class="tab-content">
                <h2>为行为添加训练视频</h2>
                <div class="form-group">
                    <label>选择行为类型</label>
                    <select id="trainBehaviorSelect">
                        <option value="">请先创建行为类型</option>
                    </select>
                </div>
                <div class="upload-area" onclick="document.getElementById('behaviorVideoInput').click()">
                    <p>📁 点击上传训练视频（支持多选）</p>
                    <p style="color: #999; margin-top: 10px;">可同时选择多个视频文件批量上传</p>
                </div>
                <input type="file" id="behaviorVideoInput" accept=".mp4,.avi,.mov,.mkv" multiple onchange="handleTrainVideo(event)">
                <div id="selectedFiles" style="margin: 15px 0; display: none;">
                    <p style="font-weight: 500;">已选择文件：</p>
                    <ul id="fileList" style="margin: 10px 0; padding-left: 20px;"></ul>
                    <button class="btn btn-success" onclick="uploadBatchVideos()">📤 开始上传</button>
                    <button class="btn btn-secondary" onclick="clearFileSelection()">取消选择</button>
                </div>
                <div class="progress" id="trainProgress">
                    <div class="progress-bar" id="trainProgressBar"></div>
                </div>
                <div class="result" id="trainResult"></div>
            </div>
        </div>
    </div>

    <script>
        let selectedFiles = [];
        
        // 页面加载时初始化
        window.onload = function() {
            loadBehaviors();
            updateTrainBehaviorSelect();
        };
        
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            
            const tabMap = {
                'detect': 0, 'stream': 1, 'behaviors': 2, 'create': 3, 'train': 4
            };
            
            document.querySelectorAll('.tab')[tabMap[tabName]].classList.add('active');
            document.getElementById(tabName + 'Tab').classList.add('active');
            
            if (tabName === 'behaviors') loadBehaviors();
            if (tabName === 'train') updateTrainBehaviorSelect();
        }
        
        // 加载行为列表
        async function loadBehaviors() {
            try {
                const response = await fetch('/behaviors/list');
                const data = await response.json();
                
                const listContainer = document.getElementById('behaviorList');
                const statsContainer = document.getElementById('behaviorStats');
                
                if (data.behaviors.length === 0) {
                    listContainer.innerHTML = '<p style="color: #999; text-align: center;">暂无行为类型，请先创建</p>';
                    statsContainer.textContent = '';
                    return;
                }
                
                statsContainer.textContent = `共 ${data.total} 个行为类型`;
                
                const categoryColors = {
                    'hygiene': '#28a745', 'service': '#007bff', 
                    'safety': '#ffc107', 'violation': '#dc3545', 'general': '#6c757d'
                };
                
                listContainer.innerHTML = data.behaviors.map(b => `
                    <div class="behavior-card" style="border-left-color: ${b.color || '#667eea'}">
                        <h3>${b.display_name}</h3>
                        <span class="category" style="background: ${categoryColors[b.category] || '#6c757d'}">
                            ${getCategoryName(b.category)}
                        </span>
                        <p style="color: #666; margin: 10px 0;">${b.description || '暂无描述'}</p>
                        <div class="video-count">📹 训练视频: ${b.video_count || 0} 个</div>
                        <div class="actions">
                            <button class="btn" style="padding: 6px 12px; font-size: 14px;" 
                                    onclick="addVideoToBehavior('${b.behavior_id}', '${b.display_name}')">
                                添加视频
                            </button>
                            <button class="btn btn-danger" style="padding: 6px 12px; font-size: 14px;" 
                                    onclick="deleteBehavior('${b.behavior_id}')">
                                删除
                            </button>
                        </div>
                    </div>
                `).join('');
                
            } catch (error) {
                console.error('加载行为列表失败:', error);
            }
        }
        
        function getCategoryName(category) {
            const names = {
                'hygiene': '卫生', 'service': '服务', 
                'safety': '安全', 'violation': '违规', 'general': '一般'
            };
            return names[category] || category;
        }
        
        // 创建行为
        async function createBehavior() {
            const formData = new FormData();
            formData.append('behavior_id', document.getElementById('behaviorId').value);
            formData.append('display_name', document.getElementById('displayName').value);
            formData.append('category', document.getElementById('category').value);
            formData.append('description', document.getElementById('description').value);
            
            const result = document.getElementById('createResult');
            result.style.display = 'none';
            
            try {
                const response = await fetch('/behaviors/create', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    result.className = 'result success';
                    result.innerHTML = `<h3>✅ 创建成功</h3><p>行为 "${data.behavior.display_name}" 已创建</p>`;
                    document.getElementById('createBehaviorForm').reset();
                } else {
                    result.className = 'result error';
                    result.innerHTML = `<h3>❌ 创建失败</h3><p>${data.message || '请检查输入'}</p>`;
                }
                result.style.display = 'block';
                
            } catch (error) {
                result.className = 'result error';
                result.innerHTML = `<h3>❌ 创建失败</h3><p>${error.message}</p>`;
                result.style.display = 'block';
            }
        }
        
        // 删除行为
        async function deleteBehavior(behaviorId) {
            if (!confirm(`确定要删除行为 "${behaviorId}" 吗？`)) return;
            
            try {
                const response = await fetch(`/behaviors/${behaviorId}`, {
                    method: 'DELETE'
                });
                
                const data = await response.json();
                if (data.success) {
                    loadBehaviors();
                    updateTrainBehaviorSelect();
                }
            } catch (error) {
                console.error('删除失败:', error);
            }
        }
        
        // 更新训练行为选择器
        async function updateTrainBehaviorSelect() {
            try {
                const response = await fetch('/behaviors/list');
                const data = await response.json();
                
                const select = document.getElementById('trainBehaviorSelect');
                if (data.behaviors.length === 0) {
                    select.innerHTML = '<option value="">请先创建行为类型</option>';
                } else {
                    select.innerHTML = data.behaviors.map(b => 
                        `<option value="${b.behavior_id}">${b.display_name}</option>`
                    ).join('');
                }
            } catch (error) {
                console.error('更新选择器失败:', error);
            }
        }
        
        // 处理检测文件
        async function handleDetectFile(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('return_all', 'true');
            
            const progress = document.getElementById('detectProgress');
            const progressBar = document.getElementById('detectProgressBar');
            const result = document.getElementById('detectResult');
            
            progress.style.display = 'block';
            progressBar.style.width = '50%';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/detect/dynamic', {
                    method: 'POST',
                    body: formData
                });
                
                progressBar.style.width = '100%';
                const data = await response.json();
                
                if (data.error) {
                    result.className = 'result error';
                    result.innerHTML = `<h3>❌ 检测失败</h3><p>${data.error}</p>`;
                } else {
                    result.className = data.is_confident ? 'result success' : 'result info';
                    result.innerHTML = `
                        <h3>${data.is_confident ? '✅' : '⚠️'} ${data.behavior_name}</h3>
                        <p>置信度: ${(data.confidence * 100).toFixed(1)}%</p>
                        <p>${data.conclusion}</p>
                    `;
                    
                    // 显示详细结果
                    if (data.all_results) {
                        showDetectionDetails(data.all_results);
                    }
                }
                result.style.display = 'block';
                
            } catch (error) {
                result.className = 'result error';
                result.innerHTML = `<h3>❌ 检测失败</h3><p>${error.message}</p>`;
                result.style.display = 'block';
            }
            
            setTimeout(() => {
                progress.style.display = 'none';
                progressBar.style.width = '0%';
            }, 1000);
        }
        
        function showDetectionDetails(results) {
            const container = document.getElementById('detectionDetails');
            container.innerHTML = '<div class="detection-result"><h3>📊 详细比对结果</h3>' + 
                results.slice(0, 5).map((r, i) => `
                    <div class="similarity-bar">
                        <div class="label">${r.behavior_name}</div>
                        <div class="bar">
                            <div class="bar-fill" style="width: ${r.similarity * 100}%; background: ${r.color || '#667eea'}"></div>
                        </div>
                        <div class="value">${(r.similarity * 100).toFixed(1)}%</div>
                    </div>
                `).join('') + '</div>';
        }
        
        // 添加视频到行为
        function addVideoToBehavior(behaviorId, behaviorName) {
            document.getElementById('trainBehaviorSelect').value = behaviorId;
            switchTab('train');
        }
        
        // 处理训练视频选择
        function handleTrainVideo(event) {
            const files = event.target.files;
            if (!files || files.length === 0) return;
            
            selectedFiles = Array.from(files);
            
            // 显示已选择的文件列表
            const fileListEl = document.getElementById('fileList');
            const selectedFilesDiv = document.getElementById('selectedFiles');
            
            fileListEl.innerHTML = selectedFiles.map((f, i) => 
                `<li>${f.name} (${(f.size / 1024 / 1024).toFixed(2)} MB)</li>`
            ).join('');
            
            selectedFilesDiv.style.display = 'block';
            document.getElementById('trainResult').style.display = 'none';
        }
        
        function clearFileSelection() {
            selectedFiles = [];
            document.getElementById('behaviorVideoInput').value = '';
            document.getElementById('selectedFiles').style.display = 'none';
            document.getElementById('trainResult').style.display = 'none';
        }
        
        // 批量上传视频
        async function uploadBatchVideos() {
            const behaviorId = document.getElementById('trainBehaviorSelect').value;
            if (!behaviorId) {
                alert('请先选择行为类型');
                return;
            }
            
            if (selectedFiles.length === 0) {
                alert('请先选择视频文件');
                return;
            }
            
            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('files', file);
            });
            
            const progress = document.getElementById('trainProgress');
            const progressBar = document.getElementById('trainProgressBar');
            const result = document.getElementById('trainResult');
            
            progress.style.display = 'block';
            progressBar.style.width = '30%';
            result.style.display = 'none';
            
            try {
                progressBar.style.width = '50%';
                
                const response = await fetch(`/behaviors/${behaviorId}/add_videos_batch`, {
                    method: 'POST',
                    body: formData
                });
                
                progressBar.style.width = '100%';
                const data = await response.json();
                
                if (data.success) {
                    result.className = 'result success';
                    result.innerHTML = `
                        <h3>✅ 批量上传完成</h3>
                        <p>成功: ${data.success_count} 个，失败: ${data.failed_count} 个</p>
                        <p>行为 "${behaviorId}" 现有 ${data.video_count} 个训练视频</p>
                        ${data.failed_count > 0 ? '<details><summary>失败详情</summary><ul>' + 
                            data.results.filter(r => !r.success).map(r => 
                                `<li>${r.filename}: ${r.error}</li>`
                            ).join('') + '</ul></details>' : ''}
                    `;
                    // 清空选择
                    clearFileSelection();
                    // 刷新行为列表
                    loadBehaviors();
                } else {
                    result.className = 'result error';
                    result.innerHTML = `<h3>❌ 上传失败</h3>`;
                }
                result.style.display = 'block';
                
            } catch (error) {
                result.className = 'result error';
                result.innerHTML = `<h3>❌ 上传失败</h3><p>${error.message}</p>`;
                result.style.display = 'block';
            }
            
            setTimeout(() => {
                progress.style.display = 'none';
                progressBar.style.width = '0%';
            }, 1000);
        }
        
        // 实时流检测
        let streamInterval = null;
        
        async function startStream() {
            const statusDiv = document.getElementById('streamStatus');
            const resultDiv = document.getElementById('streamResult');
            
            statusDiv.className = 'result info';
            statusDiv.innerHTML = '<p>🔄 正在启动摄像头...</p>';
            
            try {
                const response = await fetch('/stream/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({source: '0'})
                });
                
                const data = await response.json();
                
                if (data.success) {
                    statusDiv.className = 'result success';
                    statusDiv.innerHTML = '<p>✅ 摄像头已启动，正在检测中...</p>';
                    
                    // 开始轮询结果
                    streamInterval = setInterval(pollStreamResult, 2000);
                } else {
                    statusDiv.className = 'result error';
                    statusDiv.innerHTML = `<p>❌ 启动失败: ${data.message || '未知错误'}</p>`;
                }
            } catch (error) {
                statusDiv.className = 'result error';
                statusDiv.innerHTML = `<p>❌ 启动失败: ${error.message}</p>`;
            }
        }
        
        async function stopStream() {
            const statusDiv = document.getElementById('streamStatus');
            
            try {
                await fetch('/stream/stop', {method: 'POST'});
                
                if (streamInterval) {
                    clearInterval(streamInterval);
                    streamInterval = null;
                }
                
                statusDiv.className = 'result info';
                statusDiv.innerHTML = '<p>⏹️ 检测已停止</p>';
                document.getElementById('streamResult').innerHTML = '';
            } catch (error) {
                statusDiv.className = 'result error';
                statusDiv.innerHTML = `<p>❌ 停止失败: ${error.message}</p>`;
            }
        }
        
        async function pollStreamResult() {
            const resultDiv = document.getElementById('streamResult');
            
            try {
                const response = await fetch('/stream/result');
                const data = await response.json();
                
                if (data.status === 'not_started') {
                    return;
                }
                
                if (data.is_mopping !== undefined) {
                    const isMopping = data.is_mopping;
                    const similarity = data.mop_similarity || 0;
                    
                    resultDiv.innerHTML = `
                        <div class="result ${isMopping ? 'success' : 'error'}" style="display: block;">
                            <h3>${isMopping ? '✅ 检测到拖地行为' : '❌ 未检测到拖地行为'}</h3>
                            <p>时间: ${data.timestamp || new Date().toLocaleString()}</p>
                            <p>相似度: ${(similarity * 100).toFixed(1)}%</p>
                        </div>
                    `;
                }
            } catch (error) {
                console.error('获取流结果失败:', error);
            }
        }
    </script>
</body>
</html>
    """
    
    # 保存 HTML 文件
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return index_path


# ==================== 启动函数 ====================

def start_api():
    """启动 API 服务"""
    # 确保静态文件存在
    if config.API_WEB_UI_ENABLED:
        create_simple_ui()
    
    logger.info(f"启动 API 服务: http://{config.API_HOST}:{config.API_PORT}")
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )


if __name__ == "__main__":
    start_api()
