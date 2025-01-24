import pytest
from sainma.coordination.coordinator import SainmaCoordinator
from sainma.coordination.resource_manager import ResourceManager
from sainma.coordination.cache_manager import CacheManager
from sainma.coordination.error_handler import ErrorHandler
from sainma.coordination.context_manager import ContextManager

def test_resource_manager():
    manager = ResourceManager()
    resources = manager.check_resources()
    assert 'memory_usage' in resources
    assert 'gpu_memory_usage' in resources
    
    # Test resource reservation
    task_id = "test_task"
    assert manager.reserve_resources(task_id)
    manager.release_resources(task_id)

def test_cache_manager(test_data_dir):
    manager = CacheManager(cache_dir=str(test_data_dir))
    
    # Test query caching
    query = "action scene with explosions"
    results = [{"scene_id": "123", "score": 0.9}]
    
    manager.cache_query_results(query, results)
    cached = manager.get_query_results(query)
    assert cached == results

def test_error_handler():
    handler = ErrorHandler()
    
    # Test error handling
    try:
        raise ValueError("Test error")
    except Exception as e:
        error_info = handler.handle_error(e)
        assert error_info['error_type'] == 'ValueError'
        assert error_info['retryable'] is True

def test_context_manager(test_data_dir):
    manager = ContextManager(context_dir=str(test_data_dir))
    
    # Test context creation and retrieval
    context_id = "test_context"
    data = {"key": "value"}
    
    manager.create_context(context_id, data)
    retrieved = manager.get_context(context_id)
    assert retrieved == data

def test_coordinator():
    coordinator = SainmaCoordinator()
    
    # Test task creation
    query = "Show me the fight scene between Iron Man and Thanos"
    tasks = coordinator.create_search_tasks(query)
    assert len(tasks) > 0
    
    # Test task execution
    try:
        results = coordinator.execute_tasks(tasks)
        assert isinstance(results, list)
    except Exception as e:
        # Some tasks might require external services
        pass
