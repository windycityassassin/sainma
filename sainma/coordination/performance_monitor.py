"""Performance monitoring module for Sainma."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import psutil
import numpy as np

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_io: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    frame_rate: Optional[float] = None
    queue_size: int = 0

class PerformanceMonitor:
    """Monitors system and application performance."""
    
    def __init__(self, sample_interval: float = 1.0):
        """Initialize performance monitor."""
        self.sample_interval = sample_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self._start_time = time.time()
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self._start_time = time.time()
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        metrics = PerformanceMetrics(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            disk_io={
                'read_bytes': psutil.disk_io_counters().read_bytes,
                'write_bytes': psutil.disk_io_counters().write_bytes
            },
            processing_time=time.time() - self._start_time
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_average_metrics(self, window: int = 10) -> PerformanceMetrics:
        """Get average metrics over the last n samples."""
        if not self.metrics_history:
            return PerformanceMetrics()
        
        # Get last n samples
        samples = self.metrics_history[-window:]
        
        # Calculate averages
        avg_metrics = PerformanceMetrics(
            cpu_percent=np.mean([m.cpu_percent for m in samples]),
            memory_percent=np.mean([m.memory_percent for m in samples]),
            disk_io={
                'read_bytes': np.mean([m.disk_io['read_bytes'] for m in samples]),
                'write_bytes': np.mean([m.disk_io['write_bytes'] for m in samples])
            },
            processing_time=np.mean([m.processing_time for m in samples])
        )
        
        # Calculate frame rate if available
        frame_rates = [m.frame_rate for m in samples if m.frame_rate is not None]
        if frame_rates:
            avg_metrics.frame_rate = np.mean(frame_rates)
        
        return avg_metrics
    
    def reset(self):
        """Reset the monitor."""
        self.metrics_history.clear()
        self._start_time = time.time()
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get a summary of performance metrics."""
        if not self.metrics_history:
            return {}
        
        avg_metrics = self.get_average_metrics()
        return {
            'avg_cpu_percent': avg_metrics.cpu_percent,
            'avg_memory_percent': avg_metrics.memory_percent,
            'avg_disk_read_bytes': avg_metrics.disk_io['read_bytes'],
            'avg_disk_write_bytes': avg_metrics.disk_io['write_bytes'],
            'avg_processing_time': avg_metrics.processing_time,
            'avg_frame_rate': avg_metrics.frame_rate if avg_metrics.frame_rate else 0.0
        }
