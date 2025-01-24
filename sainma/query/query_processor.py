"""Query processing module."""

from typing import Dict, Any, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from sainma.utils.logger import get_logger

logger = get_logger(__name__)

class QueryProcessor:
    """Processes natural language queries for movie scene searches."""
    
    def __init__(self):
        """Initialize the query processor."""
        # Load language model for NLP tasks
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load sentence transformer for semantic search
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define common intents and their keywords
        self.intent_keywords = {
            'action_sequence': ['fight', 'chase', 'battle', 'action', 'fight scene'],
            'dialogue': ['conversation', 'talk', 'say', 'discuss', 'speaking'],
            'emotional': ['happy', 'sad', 'angry', 'emotional', 'crying', 'laughing'],
            'location': ['at', 'in', 'where', 'location', 'place'],
            'character_focused': ['character', 'who', 'person', 'appears']
        }
        
        # Define temporal relation keywords
        self.temporal_keywords = {
            'before': ['before', 'prior to', 'earlier'],
            'after': ['after', 'following', 'later'],
            'during': ['during', 'while', 'when']
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query."""
        # Parse query with spaCy
        doc = self.nlp(query.lower())
        
        # Extract components
        result = {
            'query': query,
            'intent': self._extract_intent(doc),
            'characters': self._extract_characters(doc),
            'actions': self._extract_actions(doc),
            'scene_types': self._extract_scene_types(doc),
            'emotion': self._extract_emotion(doc),
            'temporal_context': self._extract_temporal_context(doc),
            'embedding': None  # Will be set by get_query_embedding
        }
        
        # Generate embedding
        result['embedding'] = self.get_query_embedding(query)
        
        return result
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for the query."""
        return self.encoder.encode(query)
    
    def _extract_intent(self, doc) -> str:
        """Extract the primary intent from the query."""
        text = doc.text.lower()
        
        # Check each intent's keywords
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in text for keyword in keywords):
                return intent
        
        # Default to character_focused if no clear intent
        return 'character_focused'
    
    def _extract_characters(self, doc) -> List[str]:
        """Extract character names from the query."""
        characters = []
        
        # Look for proper nouns (potential character names)
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                characters.append(ent.text)
        
        # Also look for capitalized words not at the start of sentences
        for token in doc[1:]:  # Skip first token
            if token.text[0].isupper() and not token.is_sent_start:
                characters.append(token.text)
        
        return list(set(characters))
    
    def _extract_actions(self, doc) -> List[str]:
        """Extract action verbs from the query."""
        actions = []
        
        for token in doc:
            # Look for verbs
            if token.pos_ == 'VERB':
                actions.append(token.lemma_)
            # Also include action-related nouns
            elif token.pos_ == 'NOUN' and any(
                keyword in token.text.lower() 
                for intent in ['action_sequence']
                for keyword in self.intent_keywords[intent]
            ):
                actions.append(token.text)
        
        return list(set(actions))
    
    def _extract_scene_types(self, doc) -> List[str]:
        """Extract types of scenes from the query."""
        scene_types = []
        text = doc.text.lower()
        
        # Common scene type keywords
        scene_keywords = {
            'action': ['action', 'fight', 'chase', 'battle'],
            'dialogue': ['conversation', 'talking', 'discussion'],
            'emotional': ['emotional', 'dramatic', 'intense'],
            'montage': ['montage', 'sequence', 'compilation']
        }
        
        for scene_type, keywords in scene_keywords.items():
            if any(keyword in text for keyword in keywords):
                scene_types.append(scene_type)
        
        return scene_types
    
    def _extract_emotion(self, doc) -> Optional[str]:
        """Extract emotional context from the query."""
        text = doc.text.lower()
        
        # Common emotions and their keywords
        emotion_keywords = {
            'happy': ['happy', 'joyful', 'cheerful', 'excited'],
            'sad': ['sad', 'depressed', 'crying', 'upset'],
            'angry': ['angry', 'furious', 'mad', 'rage'],
            'scared': ['scared', 'frightened', 'terrified'],
            'surprised': ['surprised', 'shocked', 'amazed']
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text for keyword in keywords):
                return emotion
        
        return None
    
    def _extract_temporal_context(self, doc) -> Dict[str, Any]:
        """Extract temporal relations from the query."""
        text = doc.text.lower()
        
        for relation, keywords in self.temporal_keywords.items():
            if any(keyword in text for keyword in keywords):
                # Try to find the reference event
                # This is a simple implementation - could be made more sophisticated
                for token in doc:
                    if token.dep_ in ['nsubj', 'dobj'] and token.pos_ == 'NOUN':
                        return {
                            'relation': relation,
                            'reference_event': token.text
                        }
                return {'relation': relation, 'reference_event': None}
        
        return {'relation': None, 'reference_event': None}
