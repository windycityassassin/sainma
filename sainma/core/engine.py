"""
Main engine for the Sainma system
"""

from typing import Dict, Any, Optional
from sainma.agents.chief_agent import ChiefAgent
from sainma.agents.movie_expert import MovieExpertAgent
from sainma.agents.visual_analyst import VisualAnalystAgent
from sainma.agents.dialogue_expert import DialogueExpertAgent
from sainma.agents.clip_director import ClipDirectorAgent
from crewai import Crew, Task
import asyncio

class SainmaEngine:
    """
    Main engine class that coordinates all components of the Sainma system
    """
    
    def __init__(self, openai_api_key: str):
        """Initialize the Sainma engine"""
        self.openai_api_key = openai_api_key
        self.chief_agent = ChiefAgent(openai_api_key)
        self.movie_expert = MovieExpertAgent(openai_api_key)
        self.visual_analyst = VisualAnalystAgent(openai_api_key)
        self.dialogue_expert = DialogueExpertAgent(openai_api_key)
        self.clip_director = ClipDirectorAgent(openai_api_key)
        self.crew = None
        
    def _create_crew(self) -> Crew:
        """Create the CrewAI crew with all agents"""
        return Crew(
            agents=[
                self.chief_agent.agent,
                self.movie_expert.agent,
                self.visual_analyst.agent,
                self.dialogue_expert.agent,
                self.clip_director.agent
            ],
            tasks=[],  # Tasks will be created dynamically based on queries
            verbose=True
        )
    
    async def process_query(self, query: str, video_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query about a movie
        
        Args:
            query: The user's question or request
            video_path: Optional path to a video file if the query involves video analysis
            
        Returns:
            Dictionary containing the response and any generated content
        """
        # Create a new crew for this query
        self.crew = self._create_crew()
        
        # Create tasks based on the query
        tasks = self._create_tasks(query, video_path)
        self.crew.tasks = tasks
        
        # Execute the tasks
        result = await self.crew.run()
        return self._process_result(result)
    
    def _create_tasks(self, query: str, video_path: Optional[str] = None) -> list[Task]:
        """Create tasks based on the query"""
        tasks = []
        
        # Add query understanding task for chief agent
        tasks.append(
            Task(
                description=f"Understand and plan the execution of the query: {query}",
                agent=self.chief_agent.agent
            )
        )
        
        # TODO: Add more task creation logic based on query type
        return tasks
    
    def _process_result(self, result: Any) -> Dict[str, Any]:
        """Process and format the crew's result"""
        # TODO: Implement result processing logic
        return {
            "response": result,
            "metadata": {}
        }
