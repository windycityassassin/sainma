from typing import Dict, Any, Optional, Callable, List, Type, Union
import time
from dataclasses import dataclass
from enum import Enum
import threading
from functools import wraps
import traceback
from sainma.utils.logger import get_logger

logger = get_logger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"          # Can continue with degraded functionality
    MEDIUM = "medium"    # Requires retry
    HIGH = "high"       # Requires immediate attention
    CRITICAL = "critical"  # System cannot continue

class ErrorCategory(Enum):
    RESOURCE = "resource"  # Resource-related errors
    MODEL = "model"        # ML model errors
    NETWORK = "network"    # Network-related errors
    DATA = "data"         # Data processing errors
    SYSTEM = "system"     # System-level errors

@dataclass
class ErrorConfig:
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay between retries in seconds
    max_delay: float = 30.0  # Maximum delay between retries
    exponential_base: float = 2.0  # Base for exponential backoff
    circuit_window_seconds: int = 60  # Circuit breaker window in seconds
    error_threshold: int = 5  # Number of errors within the window to break the circuit

class ResourceError(Exception):
    """Error related to resource management."""
    pass

class CacheError(Exception):
    """Error related to cache operations."""
    pass

class AgentError(Exception):
    """Error related to AI agent operations."""
    pass

class VideoProcessingError(Exception):
    """Error related to video processing."""
    pass

class SearchError(Exception):
    """Error related to search operations."""
    pass

class ErrorHandler:
    """Handles errors and implements recovery strategies."""
    
    def __init__(self, config: Optional[ErrorConfig] = None):
        """Initialize error handler."""
        self.config = config or ErrorConfig()
        self._error_counts: Dict[str, int] = {}
        self._error_timestamps: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle an error and return error information."""
        error_info = {
            'error': str(error),
            'error_type': type(error).__name__,
            'traceback': traceback.format_exc(),
            'severity': severity.value,
            'category': category.value,
            'context': context or {},
            'task_id': task_id,
            'timestamp': time.time()
        }
        
        logger.error(f"Error occurred: {error_info}")
        
        # Update error counts and timestamps
        error_key = f"{category.value}:{type(error).__name__}"
        with self._lock:
            self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
            if error_key not in self._error_timestamps:
                self._error_timestamps[error_key] = []
            self._error_timestamps[error_key].append(time.time())
        
        # Check circuit breaker
        if self._should_break_circuit(error_key):
            logger.warning(f"Circuit breaker triggered for {error_key}")
            return error_info
        
        # Get and execute recovery strategy
        if self._is_retryable(error):
            strategy = self._get_recovery_strategy(error, category)
            if strategy:
                try:
                    self._execute_recovery_strategy(strategy, context)
                except Exception as e:
                    logger.error(f"Recovery strategy failed: {str(e)}")
                    error_info['recovery_error'] = str(e)
        
        return error_info
    
    def _is_retryable(self, error: Exception) -> bool:
        """Determine if an error is retryable."""
        non_retryable = (KeyboardInterrupt, SystemExit, MemoryError)
        return not isinstance(error, non_retryable)
    
    def _should_break_circuit(self, error_key: str) -> bool:
        """Implement circuit breaker pattern."""
        with self._lock:
            if error_key not in self._error_timestamps:
                return False
            
            # Get timestamps within window
            now = time.time()
            window_start = now - self.config.circuit_window_seconds
            recent_errors = [
                t for t in self._error_timestamps[error_key]
                if t >= window_start
            ]
            
            # Update timestamps list
            self._error_timestamps[error_key] = recent_errors
            
            return len(recent_errors) >= self.config.error_threshold
    
    def _get_recovery_strategy(
        self,
        error: Exception,
        category: ErrorCategory
    ) -> Optional[str]:
        """Get appropriate recovery strategy based on error type and category."""
        strategies = {
            ErrorCategory.RESOURCE: '_optimize_resources',
            ErrorCategory.DATA: '_clear_cache',
            ErrorCategory.MODEL: '_restart_agent',
            ErrorCategory.NETWORK: None,  # Let retry mechanism handle it
            ErrorCategory.SYSTEM: None    # System errors may need manual intervention
        }
        
        return strategies.get(category)
    
    def _execute_recovery_strategy(
        self,
        strategy: str,
        context: Optional[Dict[str, Any]]
    ):
        """Execute the specified recovery strategy."""
        if hasattr(self, strategy):
            strategy_func = getattr(self, strategy)
            strategy_func(context)
    
    def _optimize_resources(self):
        """Optimize system resources."""
        # Implementation would depend on specific resource management needs
        pass
    
    def _clear_cache(self):
        """Clear system cache."""
        # Implementation would depend on caching system
        pass
    
    def _restart_agent(self, context: Optional[Dict[str, Any]]):
        """Restart the specified agent."""
        # Implementation would depend on agent management system
        pass
    
    def _retry_video_processing(self, context: Optional[Dict[str, Any]]):
        """Retry video processing with different parameters."""
        # Implementation would depend on video processing system
        pass
    
    def _fallback_search(self, context: Optional[Dict[str, Any]]):
        """Use fallback search strategy."""
        # Implementation would depend on search system
        pass
    
    def should_retry(self, error_info: Dict[str, Any], attempt: int) -> bool:
        """Determine if an operation should be retried."""
        if attempt >= self.config.max_retries:
            return False
        
        severity = ErrorSeverity(error_info['severity'])
        return severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]
    
    def get_retry_delay(self, attempt: int) -> float:
        """Calculate delay before next retry using exponential backoff."""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        return min(delay, self.config.max_delay)
    
    async def retry_operation(
        self,
        operation: Callable,
        *args,
        retry_on: Optional[List[Type[Exception]]] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ):
        """Retry an operation with exponential backoff."""
        retry_on = retry_on or [Exception]
        max_retries = max_retries or self.config.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except tuple(retry_on) as e:
                if attempt == max_retries:
                    raise
                
                delay = self.get_retry_delay(attempt)
                logger.warning(
                    f"Operation failed (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying in {delay:.2f}s"
                )
                
                time.sleep(delay)
    
    @staticmethod
    def with_retry(
        retry_on: Optional[List[Type[Exception]]] = None,
        max_retries: Optional[int] = None
    ):
        """Decorator for retrying operations."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                return await self.retry_operation(
                    func,
                    self,
                    *args,
                    retry_on=retry_on,
                    max_retries=max_retries,
                    **kwargs
                )
            return wrapper
        return decorator
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            return {
                'counts': self._error_counts.copy(),
                'timestamps': {
                    k: v.copy() for k, v in self._error_timestamps.items()
                }
            }
    
    def clear_error_history(self):
        """Clear error history and counts."""
        with self._lock:
            self._error_counts.clear()
            self._error_timestamps.clear()
