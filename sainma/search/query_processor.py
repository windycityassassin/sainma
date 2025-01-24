"""Natural language query processing for Sainma."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from sainma.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class QueryComponents:
    """Components extracted from a natural language query."""
    intent: str
    entities: List[str]
    scene_types: List[str]
    emotions: List[str]
    temporal_context: Optional[Dict[str, Any]]
    embedding: torch.Tensor

class QueryProcessor:
    """Processes natural language queries into structured components."""
    
    def __init__(self):
        """Initialize the query processor."""
        # Load models
        self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", framework="pt")
        self.classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Initialize scene type patterns
        self.scene_patterns = {
            'action': r'fight|battle|chase|explosion|action',
            'dialogue': r'conversation|talk|discuss|dialogue|speaking',
            'emotional': r'cry|laugh|angry|happy|sad|emotional',
            'transition': r'montage|fade|dissolve|transition',
            'establishing': r'show|introduce|establish|begin',
        }
        
        # Initialize emotion patterns
        self.emotion_patterns = {
            'happy': r'happy|joy|celebrate|laugh|smile',
            'sad': r'sad|cry|depressed|mourn|grief',
            'angry': r'angry|rage|furious|mad|upset',
            'fear': r'scared|afraid|terrified|fear|panic',
            'surprise': r'surprised|shocked|amazed|astonished',
        }
        
        # Temporal context patterns
        self.temporal_patterns = {
            'before': r'before|prior to|earlier than',
            'after': r'after|following|later than',
            'during': r'during|while|when|as',
            'between': r'between|from.*to',
        }
    
    async def process_query(self, query: str) -> QueryComponents:
        """Process a natural language query into structured components."""
        try:
            # Clean query
            cleaned_query = self._clean_query(query)
            
            # Extract intent
            intent = await self._extract_intent(cleaned_query)
            
            # Extract entities
            entities = await self._extract_entities(cleaned_query)
            
            # Extract scene types
            scene_types = self._extract_scene_types(cleaned_query)
            
            # Extract emotions
            emotions = self._extract_emotions(cleaned_query)
            
            # Extract temporal context
            temporal_context = self._extract_temporal_context(cleaned_query)
            
            # Generate embedding
            embedding = self._generate_embedding(cleaned_query)
            
            return QueryComponents(
                intent=intent,
                entities=entities,
                scene_types=scene_types,
                emotions=emotions,
                temporal_context=temporal_context,
                embedding=embedding
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query."""
        # Convert to lowercase
        query = query.lower()
        
        # Remove special characters except basic punctuation
        query = re.sub(r'[^\w\s.,!?-]', '', query)
        
        # Normalize whitespace
        query = ' '.join(query.split())
        
        return query
    
    async def _extract_intent(self, query: str) -> str:
        """Extract the primary intent of the query."""
        try:
            # Classify intent
            result = self.classifier(query, return_all_scores=True)[0]
            
            # Get highest scoring intent
            intent = max(result, key=lambda x: x['score'])['label']
            
            return intent
            
        except Exception as e:
            logger.error(f"Error extracting intent: {str(e)}")
            return "general_search"  # Default intent
    
    async def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from the query."""
        try:
            # Get NER results
            ner_results = self.ner_pipeline(query)
            
            # Extract unique entities
            entities = []
            current_entity = []
            
            for token in ner_results:
                if token['entity'].startswith('B-'):
                    if current_entity:
                        entities.append(' '.join(current_entity))
                        current_entity = []
                    current_entity.append(token['word'])
                elif token['entity'].startswith('I-'):
                    current_entity.append(token['word'])
            
            if current_entity:
                entities.append(' '.join(current_entity))
            
            return list(set(entities))
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def _extract_scene_types(self, query: str) -> List[str]:
        """Extract scene types from the query."""
        scene_types = []
        
        for scene_type, pattern in self.scene_patterns.items():
            if re.search(pattern, query):
                scene_types.append(scene_type)
        
        return scene_types
    
    def _extract_emotions(self, query: str) -> List[str]:
        """Extract emotions from the query."""
        emotions = []
        
        for emotion, pattern in self.emotion_patterns.items():
            if re.search(pattern, query):
                emotions.append(emotion)
        
        return emotions
    
    def _extract_temporal_context(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract temporal context from the query."""
        context = {}
        
        for relation, pattern in self.temporal_patterns.items():
            match = re.search(f'({pattern})\s+(.*?)(?=\s+(?:{"|".join(self.temporal_patterns.values())})|$)', query)
            if match:
                context['relation'] = relation
                context['reference'] = match.group(2).strip()
        
        return context if context else None
    
    def _generate_embedding(self, query: str) -> torch.Tensor:
        """Generate embedding for the query."""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero embedding as fallback
            return torch.zeros(self.embedding_model.get_sentence_embedding_dimension())
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vector."""
        return self.embedding_model.get_sentence_embedding_dimension()
