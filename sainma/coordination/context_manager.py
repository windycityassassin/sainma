from typing import Dict, Any, Optional, List
import threading
from dataclasses import dataclass, field
import time
import json
from pathlib import Path
from sainma.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ContextData:
    """Container for context data with metadata."""
    data: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    source_agent: Optional[str] = None
    target_agents: List[str] = field(default_factory=list)

class ContextManager:
    """Manages shared context between agents."""
    
    def __init__(self, context_dir: str = ".context/sainma"):
        """Initialize the context manager."""
        self._lock = threading.Lock()
        self._contexts: Dict[str, ContextData] = {}
        self._context_dir = Path(context_dir)
        self._context_dir.mkdir(parents=True, exist_ok=True)
    
    def create_context(
        self,
        context_id: str,
        data: Dict[str, Any],
        source_agent: Optional[str] = None,
        target_agents: Optional[List[str]] = None
    ) -> str:
        """Create a new context."""
        with self._lock:
            if context_id in self._contexts:
                raise ValueError(f"Context {context_id} already exists")
            
            context = ContextData(
                data=data,
                source_agent=source_agent,
                target_agents=target_agents or []
            )
            
            self._contexts[context_id] = context
            self._save_context(context_id, context)
            
            logger.info(f"Created context {context_id}")
            return context_id
    
    def get_context(
        self,
        context_id: str,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get context data."""
        with self._lock:
            if context_id not in self._contexts:
                # Try to load from disk
                self._load_context(context_id)
            
            context = self._contexts[context_id]
            
            # Check access permissions
            if agent_id and context.target_agents:
                if agent_id not in context.target_agents:
                    raise PermissionError(
                        f"Agent {agent_id} not authorized to access context {context_id}"
                    )
            
            # Update metadata
            context.access_count += 1
            context.updated_at = time.time()
            
            return context.data.copy()
    
    def update_context(
        self,
        context_id: str,
        updates: Dict[str, Any],
        agent_id: Optional[str] = None
    ):
        """Update existing context data."""
        with self._lock:
            if context_id not in self._contexts:
                raise KeyError(f"Context {context_id} not found")
            
            context = self._contexts[context_id]
            
            # Check update permissions
            if agent_id and context.source_agent:
                if agent_id != context.source_agent:
                    raise PermissionError(
                        f"Agent {agent_id} not authorized to update context {context_id}"
                    )
            
            # Update data
            context.data.update(updates)
            context.updated_at = time.time()
            
            # Save to disk
            self._save_context(context_id, context)
            
            logger.info(f"Updated context {context_id}")
    
    def share_context(
        self,
        context_id: str,
        target_agents: List[str],
        source_agent: Optional[str] = None
    ):
        """Share context with additional agents."""
        with self._lock:
            if context_id not in self._contexts:
                raise KeyError(f"Context {context_id} not found")
            
            context = self._contexts[context_id]
            
            # Check share permissions
            if source_agent and context.source_agent:
                if source_agent != context.source_agent:
                    raise PermissionError(
                        f"Agent {source_agent} not authorized to share context {context_id}"
                    )
            
            # Update target agents
            context.target_agents.extend(
                agent for agent in target_agents
                if agent not in context.target_agents
            )
            
            # Save updated permissions
            self._save_context(context_id, context)
            
            logger.info(f"Shared context {context_id} with {target_agents}")
    
    def _save_context(self, context_id: str, context: ContextData):
        """Save context to disk."""
        context_path = self._context_dir / f"{context_id}.json"
        with open(context_path, 'w') as f:
            json.dump({
                'data': context.data,
                'created_at': context.created_at,
                'updated_at': context.updated_at,
                'access_count': context.access_count,
                'source_agent': context.source_agent,
                'target_agents': context.target_agents
            }, f)
    
    def _load_context(self, context_id: str):
        """Load context from disk."""
        context_path = self._context_dir / f"{context_id}.json"
        if not context_path.exists():
            raise KeyError(f"Context {context_id} not found")
        
        with open(context_path, 'r') as f:
            data = json.load(f)
            self._contexts[context_id] = ContextData(**data)
    
    def get_context_metadata(self, context_id: str) -> Dict[str, Any]:
        """Get metadata about a context."""
        with self._lock:
            if context_id not in self._contexts:
                self._load_context(context_id)
            
            context = self._contexts[context_id]
            return {
                'created_at': context.created_at,
                'updated_at': context.updated_at,
                'access_count': context.access_count,
                'source_agent': context.source_agent,
                'target_agents': context.target_agents
            }
    
    def list_contexts(self) -> List[str]:
        """List all available contexts."""
        return [
            path.stem for path in self._context_dir.glob("*.json")
        ]
    
    def cleanup_old_contexts(self, max_age_seconds: float = 3600):
        """Clean up old contexts."""
        current_time = time.time()
        with self._lock:
            for context_id in list(self._contexts.keys()):
                context = self._contexts[context_id]
                if current_time - context.updated_at > max_age_seconds:
                    del self._contexts[context_id]
                    context_path = self._context_dir / f"{context_id}.json"
                    if context_path.exists():
                        context_path.unlink()
                    logger.info(f"Cleaned up old context {context_id}")
