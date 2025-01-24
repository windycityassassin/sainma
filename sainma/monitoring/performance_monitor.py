"""Performance monitoring for Sainma."""

import time
import threading
from typing import Dict, List, Optional, Any
from collections import defaultdict
import psutil
import torch
from sainma.utils.logger import get_logger

logger = get_logger(__name__)

class PerformanceMetrics:
    """Container for performance metrics."""
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.memory_usage: List[Dict[str, float]] = []
        self.gpu_usage: List[Dict[str, float]] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.cache_hits: int = 0
        self.cache_misses: int = 0

class PerformanceMonitor:
    """Monitors system performance and resource usage."""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.metrics = PerformanceMetrics()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start performance monitoring."""
        with self._lock:
            if not self._monitoring:
                self._monitoring = True
                self._monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    daemon=True
                )
                self._monitor_thread.start()
                logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        with self._lock:
            if self._monitoring:
                self._monitoring = False
                if self._monitor_thread:
                    self._monitor_thread.join()
                logger.info("Performance monitoring stopped")
    
    def record_operation_time(self, operation: str, duration: float):
        """Record the duration of an operation."""
        with self._lock:
            self.metrics.operation_times[operation].append(duration)
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        with self._lock:
            self.metrics.error_counts[error_type] += 1
    
    def record_cache_access(self, hit: bool):
        """Record a cache access."""
        with self._lock:
            if hit:
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._lock:
            metrics = {
                'operation_times': {
                    op: {
                        'mean': sum(times) / len(times) if times else 0,
                        'min': min(times) if times else 0,
                        'max': max(times) if times else 0,
                        'count': len(times)
                    }
                    for op, times in self.metrics.operation_times.items()
                },
                'memory_usage': self.metrics.memory_usage[-1] if self.metrics.memory_usage else {},
                'gpu_usage': self.metrics.gpu_usage[-1] if self.metrics.gpu_usage else {},
                'error_counts': dict(self.metrics.error_counts),
                'cache_performance': {
                    'hits': self.metrics.cache_hits,
                    'misses': self.metrics.cache_misses,
                    'hit_ratio': (
                        self.metrics.cache_hits /
                        (self.metrics.cache_hits + self.metrics.cache_misses)
                        if (self.metrics.cache_hits + self.metrics.cache_misses) > 0
                        else 0
                    )
                }
            }
            return metrics
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # CPU and memory monitoring
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                memory_metrics = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024 ** 3),
                    'memory_available_gb': memory.available / (1024 ** 3)
                }
                
                # GPU monitoring
                gpu_metrics = {}
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        memory = torch.cuda.memory_stats(i)
                        gpu_metrics[f'gpu_{i}'] = {
                            'memory_allocated_gb': memory['allocated_bytes.all'] / (1024 ** 3),
                            'memory_reserved_gb': memory['reserved_bytes.all'] / (1024 ** 3),
                            'utilization': torch.cuda.utilization(i)
                        }
                
                with self._lock:
                    self.metrics.memory_usage.append(memory_metrics)
                    if gpu_metrics:
                        self.metrics.gpu_usage.append(gpu_metrics)
                
                # Keep only recent history
                max_history = 3600  # 1 hour at 1 second intervals
                with self._lock:
                    if len(self.metrics.memory_usage) > max_history:
                        self.metrics.memory_usage = self.metrics.memory_usage[-max_history:]
                    if len(self.metrics.gpu_usage) > max_history:
                        self.metrics.gpu_usage = self.metrics.gpu_usage[-max_history:]
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
            
            time.sleep(self.sampling_interval)
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
