import psutil
import torch
from typing import Dict, Optional, List
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
from sainma.utils.logger import get_logger
import platform
import os
import gc

logger = get_logger(__name__)

@dataclass
class ResourceLimits:
    max_memory_percent: float = 80.0  # Maximum memory usage (%)
    max_gpu_memory_percent: float = 90.0  # Maximum GPU memory usage (%)
    max_concurrent_tasks: int = 3  # Maximum number of concurrent tasks
    min_free_memory_gb: float = 2.0  # Minimum free memory required (GB)

class ResourceManager:
    """Manages system resources for efficient agent operations."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        """Initialize the resource manager with specified limits."""
        self.limits = limits or ResourceLimits()
        self._lock = threading.Lock()
        self._active_tasks: Dict[str, float] = {}  # task_id: memory_reserved
        self._executor = ThreadPoolExecutor(max_workers=self.limits.max_concurrent_tasks)
        
    def check_resources(self) -> Dict[str, float]:
        """Check current resource utilization."""
        try:
            # CPU Memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_usage = memory.used / (1024 ** 3)  # Convert to GB
            free_memory_gb = memory.available / (1024 ** 3)
            
            # GPU Memory if available
            gpu_memory_percent = 0.0
            gpu_memory_usage = 0.0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated()
                gpu_total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_percent = (gpu_memory / gpu_total) * 100
                gpu_memory_usage = gpu_memory / (1024 ** 3)  # Convert to GB
            
            return {
                "memory_usage": memory_usage,
                "memory_percent": memory_percent,
                "free_memory_gb": free_memory_gb,
                "gpu_memory_usage": gpu_memory_usage,
                "gpu_memory_percent": gpu_memory_percent,
                "active_tasks": len(self._active_tasks)
            }
            
        except Exception as e:
            logger.error(f"Error checking resources: {e}")
            return {
                "memory_usage": 0.0,
                "memory_percent": 0.0,
                "free_memory_gb": 0.0,
                "gpu_memory_usage": 0.0,
                "gpu_memory_percent": 0.0,
                "active_tasks": 0
            }
    
    def can_start_task(self, required_memory_gb: float = 1.0) -> bool:
        """Check if a new task can be started based on resource availability."""
        with self._lock:
            resources = self.check_resources()
            
            # Check number of active tasks
            if len(self._active_tasks) >= self.limits.max_concurrent_tasks:
                logger.warning("Maximum concurrent tasks reached")
                return False
            
            # Check memory availability
            if resources["memory_percent"] > self.limits.max_memory_percent:
                logger.warning("Memory usage too high")
                return False
            
            if resources["free_memory_gb"] < self.limits.min_free_memory_gb:
                logger.warning("Not enough free memory")
                return False
            
            # Check GPU if available
            if torch.cuda.is_available():
                if resources["gpu_memory_percent"] > self.limits.max_gpu_memory_percent:
                    logger.warning("GPU memory usage too high")
                    return False
            
            return True
    
    def reserve_resources(self, task_id: str, required_memory_gb: float = 1.0) -> bool:
        """Reserve resources for a task."""
        with self._lock:
            if not self.can_start_task(required_memory_gb):
                return False
            
            self._active_tasks[task_id] = required_memory_gb
            logger.info(f"Reserved resources for task {task_id}")
            return True
    
    def release_resources(self, task_id: str):
        """Release resources reserved for a task."""
        with self._lock:
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]
                logger.info(f"Released resources for task {task_id}")
    
    async def optimize_memory(self):
        """Optimize memory usage by cleaning up unused resources."""
        try:
            # CPU Memory optimization
            if psutil:
                # Clear system caches
                if platform.system() == "Linux":
                    os.system("sync && echo 3 > /proc/sys/vm/drop_caches")
                elif platform.system() == "Darwin":  # macOS
                    os.system("sync && purge")
                
                # Force Python garbage collection
                gc.collect()
            
            # GPU Memory optimization
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Force garbage collection on CUDA
                gc.collect()
                torch.cuda.ipc_collect()
                
                # Check memory after cleanup
                for i in range(torch.cuda.device_count()):
                    memory = torch.cuda.memory_stats(i)
                    logger.info(
                        f"GPU {i} memory after cleanup: "
                        f"allocated={memory['allocated_bytes.all'] / 1e9:.2f}GB, "
                        f"reserved={memory['reserved_bytes.all'] / 1e9:.2f}GB"
                    )
            
            logger.info("Memory optimization completed")
            
        except Exception as e:
            logger.error(f"Error during memory optimization: {str(e)}")
            raise
    
    def get_task_allocation(self) -> Dict[str, List[str]]:
        """Get current task allocation across available resources."""
        with self._lock:
            allocation = {
                "cpu": [],
                "gpu": [] if torch.cuda.is_available() else None,
                "memory": [f"{task}: {mem:.2f}GB" for task, mem in self._active_tasks.items()]
            }
            return allocation
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._executor.shutdown(wait=True)
        import asyncio
        asyncio.run(self.optimize_memory())
