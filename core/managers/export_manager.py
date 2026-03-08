# export_manager.py
import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from core.utils.config import MoppingDetectionConfig
from core.managers.logger import get_logger

logger = get_logger(__name__)


class ExportManager:
    """检测结果导出管理器（支持 CSV/JSON 格式）"""
    
    def __init__(self, config: MoppingDetectionConfig = None):
        self.config = config or MoppingDetectionConfig()
        self.export_dir = self.config.EXPORT_OUTPUT_DIR
        self._ensure_export_dir()
    
    def _ensure_export_dir(self):
        """确保导出目录存在"""
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir, exist_ok=True)
            logger.info(f"创建导出目录: {self.export_dir}")
    
    def _generate_filename(self, prefix: str = "detection", extension: str = "csv") -> str:
        """生成导出文件名"""
        if self.config.EXPORT_INCLUDE_TIMESTAMP:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.{extension}"
        else:
            filename = f"{prefix}.{extension}"
        return os.path.join(self.export_dir, filename)
    
    def export_to_csv(self, results: List[Dict[str, Any]], filename: str = None) -> str:
        """导出检测结果为 CSV 格式"""
        if filename is None:
            filename = self._generate_filename("detection", "csv")
        
        # 定义 CSV 列
        fieldnames = [
            'id', 'video_path', 'is_mopping', 'mop_similarity', 
            'non_mop_similarity', 'similarity_diff', 'confidence',
            'method', 'threshold', 'timestamp'
        ]
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for i, result in enumerate(results):
                    row = {
                        'id': result.get('id', i),
                        'video_path': result.get('video_path', ''),
                        'is_mopping': result.get('is_mopping', False),
                        'mop_similarity': round(result.get('mop_similarity', 0), 4),
                        'non_mop_similarity': round(result.get('non_mop_similarity', 0), 4),
                        'timestamp': result.get('timestamp', datetime.now().isoformat())
                    }
                    
                    # 添加详细信息
                    details = result.get('details', {})
                    row['similarity_diff'] = details.get('similarity_diff', 0)
                    row['confidence'] = details.get('confidence', 0)
                    row['method'] = details.get('method', 'cosine')
                    row['threshold'] = details.get('threshold', self.config.MOPPING_THRESHOLD)
                    
                    writer.writerow(row)
            
            logger.info(f"CSV 导出成功: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"CSV 导出失败: {e}")
            raise
    
    def export_to_json(self, results: List[Dict[str, Any]], filename: str = None) -> str:
        """导出检测结果为 JSON 格式"""
        if filename is None:
            filename = self._generate_filename("detection", "json")
        
        export_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_count': len(results),
                'config': {
                    'model': self.config.MODEL_BACKBONE,
                    'threshold': self.config.MOPPING_THRESHOLD,
                    'similarity_gap': self.config.SIMILARITY_GAP,
                    'similarity_method': self.config.SIMILARITY_METHOD
                }
            },
            'results': results
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON 导出成功: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"JSON 导出失败: {e}")
            raise
    
    def export(self, results: List[Dict[str, Any]], format: str = None) -> List[str]:
        """
        导出检测结果
        
        Args:
            results: 检测结果列表
            format: 导出格式 (csv / json / both)，默认使用配置值
            
        Returns:
            导出的文件路径列表
        """
        format = format or self.config.EXPORT_FORMAT
        exported_files = []
        
        if format in ['csv', 'both']:
            csv_file = self.export_to_csv(results)
            exported_files.append(csv_file)
        
        if format in ['json', 'both']:
            json_file = self.export_to_json(results)
            exported_files.append(json_file)
        
        return exported_files
    
    def load_from_csv(self, filename: str) -> List[Dict[str, Any]]:
        """从 CSV 文件加载检测结果"""
        results = []
        try:
            with open(filename, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 转换数据类型
                    row['id'] = int(row['id'])
                    row['is_mopping'] = row['is_mopping'].lower() == 'true'
                    row['mop_similarity'] = float(row['mop_similarity'])
                    row['non_mop_similarity'] = float(row['non_mop_similarity'])
                    row['similarity_diff'] = float(row['similarity_diff'])
                    row['confidence'] = float(row['confidence'])
                    row['threshold'] = float(row['threshold'])
                    
                    # 重构 details
                    row['details'] = {
                        'similarity_diff': row.pop('similarity_diff'),
                        'confidence': row.pop('confidence'),
                        'method': row.pop('method'),
                        'threshold': row.pop('threshold')
                    }
                    
                    results.append(row)
            
            logger.info(f"从 CSV 加载 {len(results)} 条记录: {filename}")
            return results
            
        except Exception as e:
            logger.error(f"CSV 加载失败: {e}")
            raise
    
    def load_from_json(self, filename: str) -> List[Dict[str, Any]]:
        """从 JSON 文件加载检测结果"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            logger.info(f"从 JSON 加载 {len(results)} 条记录: {filename}")
            return results
            
        except Exception as e:
            logger.error(f"JSON 加载失败: {e}")
            raise


class DetectionLogger:
    """检测日志记录器（实时追加模式）"""
    
    def __init__(self, config: MoppingDetectionConfig = None):
        self.config = config or MoppingDetectionConfig()
        self.export_manager = ExportManager(config)
        self.buffer = []
        self.auto_flush_threshold = 10  # 自动刷新阈值
    
    def log(self, result: Dict[str, Any]):
        """记录单条检测结果"""
        self.buffer.append(result)
        
        # 达到阈值时自动导出
        if len(self.buffer) >= self.auto_flush_threshold:
            self.flush()
    
    def flush(self, format: str = 'csv'):
        """将缓冲区数据导出到文件"""
        if not self.buffer:
            return
        
        try:
            if format == 'csv':
                # 追加模式写入 CSV
                filename = os.path.join(
                    self.config.EXPORT_OUTPUT_DIR, 
                    "detection_log.csv"
                )
                
                file_exists = os.path.exists(filename)
                fieldnames = [
                    'id', 'video_path', 'is_mopping', 'mop_similarity', 
                    'non_mop_similarity', 'similarity_diff', 'confidence',
                    'method', 'threshold', 'timestamp'
                ]
                
                with open(filename, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    
                    for result in self.buffer:
                        row = self._format_row(result)
                        writer.writerow(row)
                
                logger.info(f"日志已追加到: {filename} ({len(self.buffer)} 条)")
            
            # 清空缓冲区
            self.buffer.clear()
            
        except Exception as e:
            logger.error(f"日志导出失败: {e}")
    
    def _format_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化结果为行数据"""
        details = result.get('details', {})
        return {
            'id': result.get('id', 0),
            'video_path': result.get('video_path', ''),
            'is_mopping': result.get('is_mopping', False),
            'mop_similarity': round(result.get('mop_similarity', 0), 4),
            'non_mop_similarity': round(result.get('non_mop_similarity', 0), 4),
            'similarity_diff': details.get('similarity_diff', 0),
            'confidence': details.get('confidence', 0),
            'method': details.get('method', 'cosine'),
            'threshold': details.get('threshold', self.config.MOPPING_THRESHOLD),
            'timestamp': result.get('timestamp', datetime.now().isoformat())
        }
    
    def close(self):
        """关闭日志记录器，确保所有数据写入"""
        self.flush()
