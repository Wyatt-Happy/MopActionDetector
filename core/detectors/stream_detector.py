# stream_detector.py
import cv2
import time
import threading
import queue
from typing import Optional, Callable, Dict, Any, Union
from collections import deque
from datetime import datetime
from core.utils.config import MoppingDetectionConfig
from core.detectors.action_detector import MoppingActionDetector
from core.managers.logger import get_logger

logger = get_logger(__name__)


class StreamBuffer:
    """视频流缓冲区"""
    
    def __init__(self, max_size: int = 10):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def add(self, frame):
        """添加帧到缓冲区"""
        with self.lock:
            self.buffer.append(frame)
    
    def get_recent(self, n: int = 1):
        """获取最近的 n 帧"""
        with self.lock:
            if len(self.buffer) == 0:
                return []
            return list(self.buffer)[-n:]
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()


class VideoStreamDetector:
    """实时视频流检测器（支持摄像头/RTMP/RTSP）"""
    
    def __init__(self, config: MoppingDetectionConfig = None):
        self.config = config or MoppingDetectionConfig()
        self.detector = MoppingActionDetector(config)
        self.buffer = StreamBuffer(max_size=self.config.STREAM_BUFFER_SIZE)
        self.is_running = False
        self.capture_thread = None
        self.detection_thread = None
        self.result_queue = queue.Queue()
        self.frame_count = 0
        self.last_detection_time = 0
        
    def start_capture(self, source: Union[int, str] = 0):
        """开始捕获视频流"""
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            logger.error(f"无法打开视频源: {source}")
            return False
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        logger.info(f"视频流捕获已启动: {source}")
        return True
    
    def _capture_loop(self):
        """视频捕获循环"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("视频流读取失败，尝试重连...")
                time.sleep(1)
                continue
            
            self.buffer.add(frame)
            self.frame_count += 1
            
            # 可选：实时显示
            if self.config.STREAM_SHOW_PREVIEW:
                display_frame = frame.copy()
                # 添加状态信息
                cv2.putText(display_frame, f"Frames: {self.frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Stream Preview", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
    
    def _detection_loop(self):
        """检测循环"""
        import tempfile
        import os
        
        while self.is_running:
            current_time = time.time()
            
            # 按间隔执行检测
            if current_time - self.last_detection_time < self.config.STREAM_SAMPLE_INTERVAL:
                time.sleep(0.1)
                continue
            
            # 获取缓冲区中的帧
            frames = self.buffer.get_recent(self.config.SAMPLE_FRAMES)
            if len(frames) < self.config.SAMPLE_FRAMES // 2:
                logger.debug("缓冲区帧数不足，跳过本次检测")
                continue
            
            # 保存临时视频文件
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    temp_path = tmp.name
                
                # 写入临时视频
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_path, fourcc, 10, (width, height))
                for frame in frames:
                    out.write(frame)
                out.release()
                
                # 执行检测
                result = self.detector.detect(temp_path)
                self.result_queue.put({
                    'timestamp': datetime.now().isoformat(),
                    'is_mopping': result[0],
                    'mop_similarity': result[1],
                    'non_mop_similarity': result[2],
                    'details': result[3]
                })
                
                self.last_detection_time = current_time
                
            except Exception as e:
                logger.error(f"检测过程出错: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
    
    def get_latest_result(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """获取最新检测结果"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """停止检测"""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("视频流检测已停止")


class ParallelVideoProcessor:
    """并行视频处理器"""
    
    def __init__(self, config: MoppingDetectionConfig = None):
        self.config = config or MoppingDetectionConfig()
        self.detector = MoppingActionDetector(config)
        self.task_queue = queue.Queue(maxsize=self.config.PARALLEL_QUEUE_SIZE)
        self.result_queue = queue.Queue()
        self.workers = []
        self.is_running = False
        
    def start(self):
        """启动工作线程"""
        self.is_running = True
        num_workers = self.config.PARALLEL_WORKERS
        if num_workers <= 0:
            import multiprocessing
            num_workers = multiprocessing.cpu_count()
        
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"并行处理器已启动，工作线程数: {num_workers}")
    
    def _worker_loop(self, worker_id: int):
        """工作线程循环"""
        logger.debug(f"工作线程 {worker_id} 已启动")
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=1)
                video_path = task['video_path']
                task_id = task.get('id', 0)
                
                logger.info(f"工作线程 {worker_id} 处理任务 {task_id}: {video_path}")
                
                # 执行检测
                result = self.detector.detect(video_path)
                
                self.result_queue.put({
                    'id': task_id,
                    'video_path': video_path,
                    'is_mopping': result[0],
                    'mop_similarity': result[1],
                    'non_mop_similarity': result[2],
                    'details': result[3]
                })
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"工作线程 {worker_id} 处理出错: {e}")
    
    def submit(self, video_paths: list) -> list:
        """提交视频任务并等待完成"""
        if not self.is_running:
            self.start()
        
        # 提交任务
        for i, path in enumerate(video_paths):
            self.task_queue.put({'id': i, 'video_path': path})
        
        # 等待所有任务完成
        self.task_queue.join()
        
        # 收集结果
        results = []
        for _ in range(len(video_paths)):
            try:
                result = self.result_queue.get(timeout=30)
                results.append(result)
            except queue.Empty:
                logger.error("等待结果超时")
                break
        
        # 按原始顺序排序
        results.sort(key=lambda x: x['id'])
        return results
    
    def stop(self):
        """停止处理器"""
        self.is_running = False
        for worker in self.workers:
            worker.join(timeout=2)
        logger.info("并行处理器已停止")
