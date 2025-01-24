from typing import Dict, List, Any, Optional
from crewai import Crew, Task, Process
from sainma.agents.chief_agent import ChiefAgent
from sainma.agents.movie_expert import MovieExpertAgent
from sainma.agents.visual_analyst import VisualAnalystAgent
from sainma.agents.dialogue_expert import DialogueExpertAgent
from sainma.agents.clip_director import ClipDirectorAgent
from sainma.utils.logger import get_logger
from .resource_manager import ResourceManager, ResourceLimits
from .cache_manager import CacheManager, CacheConfig
from .error_handler import ErrorHandler, ErrorConfig, ErrorSeverity, ErrorCategory
from .context_manager import ContextManager
from .performance_monitor import PerformanceMonitor
import time

logger = get_logger(__name__)

class SainmaCoordinator:
    """Coordinates the interaction between different agents in the Sainma system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the coordinator with all required agents."""
        self.config = config or {}
        # Initialize managers
        self.resource_manager = ResourceManager(
            ResourceLimits(
                max_memory_percent=80.0,
                max_gpu_memory_percent=90.0,
                max_concurrent_tasks=3,
                min_free_memory_gb=2.0
            )
        )
        
        self.cache_manager = CacheManager(
            CacheConfig(
                cache_dir=".cache/sainma",
                max_size_bytes=10 * 1024 * 1024 * 1024,  # 10GB
                ttl=24 * 60 * 60,  # 24 hours
                query_similarity_threshold=0.85
            )
        )
        
        self.error_handler = ErrorHandler(
            ErrorConfig(
                max_retries=3,
                base_delay=1.0,
                max_delay=30.0,
                exponential_base=2.0
            )
        )
        
        self.context_manager = ContextManager(
            context_dir=".context/sainma"
        )
        
        self.performance_monitor = PerformanceMonitor()
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        self.chief_agent = ChiefAgent()
        self.movie_expert = MovieExpertAgent()
        self.visual_analyst = VisualAnalystAgent()
        self.dialogue_expert = DialogueExpertAgent()
        self.clip_director = ClipDirectorAgent()
        
        # Initialize the crew with all agents
        self.crew = Crew(
            agents=[
                self.chief_agent.agent,
                self.movie_expert.agent,
                self.visual_analyst.agent,
                self.dialogue_expert.agent,
                self.clip_director.agent
            ],
            process=Process.sequential  # Tasks will be executed in sequence
        )
        
        # Task status tracking
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        
    def create_search_tasks(self, query: str) -> List[Task]:
        """Create tasks for processing a search query."""
        # Create shared context for the search
        context_id = f"search_{int(time.time())}"
        self.context_manager.create_context(
            context_id=context_id,
            data={'query': query},
            source_agent='chief_agent',
            target_agents=['movie_expert', 'visual_analyst', 'dialogue_expert']
        )
        
        tasks = [
            Task(
                description=f"Analyze the search query: '{query}' and create a detailed search plan",
                agent=self.chief_agent.agent,
                context={'query': query, 'context_id': context_id}
            ),
            Task(
                description=f"Provide movie knowledge and context for the query: '{query}'",
                agent=self.movie_expert.agent,
                context={'query': query, 'context_id': context_id}
            ),
            Task(
                description=f"Analyze visual elements related to the query: '{query}'",
                agent=self.visual_analyst.agent,
                context={'query': query, 'context_id': context_id}
            ),
            Task(
                description=f"Analyze dialogue and audio elements for query: '{query}'",
                agent=self.dialogue_expert.agent,
                context={'query': query, 'context_id': context_id}
            )
        ]
        return tasks
    
    def create_clip_tasks(self, scene_info: Dict[str, Any]) -> List[Task]:
        """Create tasks for generating a clip based on scene information."""
        tasks = [
            Task(
                description="Review scene information and create clip generation plan",
                agent=self.chief_agent.agent,
                context={"scene_info": scene_info}
            ),
            Task(
                description="Analyze visual composition for optimal clip points",
                agent=self.visual_analyst.agent,
                context={"scene_info": scene_info}
            ),
            Task(
                description="Analyze dialogue and audio cues for clip boundaries",
                agent=self.dialogue_expert.agent,
                context={"scene_info": scene_info}
            ),
            Task(
                description="Generate and optimize the final clip",
                agent=self.clip_director.agent,
                context={"scene_info": scene_info}
            )
        ]
        return tasks
    
    @ErrorHandler.with_retry(
        retry_on=[RuntimeError, ConnectionError],
        max_retries=3
    )
    async def execute_tasks(self, tasks: List[Task]) -> List[Dict[str, Any]]:
        """Execute a list of tasks and return their results."""
        try:
            # Check cache for similar queries if this is a search task
            if tasks and isinstance(tasks[0].context, dict) and 'query' in tasks[0].context:
                query = tasks[0].context['query']
                try:
                    cached_results = await self.cache_manager.get_query_results(query)
                    if cached_results:
                        logger.info(f"Using cached results for query: {query}")
                        return cached_results
                except Exception as e:
                    self.error_handler.handle_error(
                        error=e,
                        context={'operation': 'cache_lookup', 'query': query},
                        severity=ErrorSeverity.LOW,
                        category=ErrorCategory.DATA
                    )
                    # Continue without cache

            # Check resource availability
            try:
                resources = await self.resource_manager.check_resources()
                logger.info(f"Current resource usage: {resources}")
            except Exception as e:
                self.error_handler.handle_error(
                    error=e,
                    context={'operation': 'resource_check'},
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.RESOURCE
                )
                raise
            
            # Reserve resources for tasks
            for task in tasks:
                try:
                    if not await self.resource_manager.reserve_resources(task.id):
                        raise RuntimeError("Insufficient resources for task execution")
                    self.active_tasks[task.id] = task
                except Exception as e:
                    self.error_handler.handle_error(
                        error=e,
                        context={'operation': 'resource_reservation', 'task_id': task.id},
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.RESOURCE
                    )
                    raise
            
            try:
                # Execute tasks through the crew
                results = await self.crew.kickoff(tasks)
                
                # Cache results if this was a search task
                if tasks and isinstance(tasks[0].context, dict) and 'query' in tasks[0].context:
                    query = tasks[0].context['query']
                    try:
                        await self.cache_manager.cache_query_results(query, results)
                    except Exception as e:
                        self.error_handler.handle_error(
                            error=e,
                            context={'operation': 'cache_store', 'query': query},
                            severity=ErrorSeverity.LOW,
                            category=ErrorCategory.DATA
                        )
                
                return results
            except Exception as e:
                self.error_handler.handle_error(
                    error=e,
                    context={'operation': 'task_execution'},
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.SYSTEM
                )
                raise
            finally:
                # Always attempt to clean up resources
                for task in tasks:
                    try:
                        if task.id in self.active_tasks:
                            del self.active_tasks[task.id]
                            await self.resource_manager.release_resources(task.id)
                    except Exception as e:
                        self.error_handler.handle_error(
                            error=e,
                            context={'operation': 'resource_cleanup', 'task_id': task.id},
                            severity=ErrorSeverity.MEDIUM,
                            category=ErrorCategory.RESOURCE
                        )
                
                try:
                    await self.resource_manager.optimize_memory()
                except Exception as e:
                    self.error_handler.handle_error(
                        error=e,
                        context={'operation': 'memory_optimization'},
                        severity=ErrorSeverity.LOW,
                        category=ErrorCategory.RESOURCE
                    )
        except Exception as e:
            await self._handle_task_failure(tasks)
            raise
    
    async def _handle_task_failure(self, failed_tasks: List[Task]):
        """Handle task failures and implement recovery strategies."""
        for task in failed_tasks:
            if task.id in self.active_tasks:
                logger.warning(f"Task {task.id} failed, attempting recovery")
                # Implement retry logic or alternative execution paths
                # For now, just remove from active tasks
                del self.active_tasks[task.id]
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the current status of a task."""
        if task_id in self.active_tasks:
            return {
                "status": "active",
                "task": self.active_tasks[task_id]
            }
        
        for task in self.completed_tasks:
            if task.id == task_id:
                return {
                    "status": "completed",
                    "task": task
                }
        
        return {
            "status": "not_found",
            "task": None
        }
