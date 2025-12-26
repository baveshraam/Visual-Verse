"""
Relation Extractor - SMART VERSION
Uses NLP patterns to detect semantic relationships between concepts.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
import re

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class RelationType(Enum):
    """Types of relationships between concepts."""
    IS_A = "is_a"              # X is a type/subset of Y
    PART_OF = "part_of"        # X is part of Y
    USES = "uses"              # X uses/includes Y
    HAS = "has"                # X has Y
    EXAMPLE_OF = "example_of"  # X is an example of Y
    LEADS_TO = "leads_to"      # X leads to/causes Y
    RELATED_TO = "related_to"  # General relationship


@dataclass
class Relation:
    """Represents a relationship between two concepts."""
    source: str
    target: str
    relation_type: RelationType
    confidence: float
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type.value,
            "confidence": round(self.confidence, 3)
        }


class RelationExtractor:
    """
    Extracts semantic relationships from text using pattern matching.
    
    Detects patterns like:
    - "X is a subset of Y" → X IS_A Y
    - "types of X include Y, Z" → Y IS_A X, Z IS_A X
    - "X uses Y" → X USES Y
    """
    
    # Relationship patterns with their types
    PATTERNS = [
        # IS_A patterns
        (r'(\w[\w\s]+?) is a (?:subset|type|form|kind) of (\w[\w\s]+)', RelationType.IS_A),
        (r'(\w[\w\s]+?) is a specialized form of (\w[\w\s]+)', RelationType.IS_A),
        (r'(\w[\w\s]+?) is part of (\w[\w\s]+)', RelationType.PART_OF),
        
        # USES patterns
        (r'(\w[\w\s]+?) uses? (\w[\w\s]+)', RelationType.USES),
        (r'(\w[\w\s]+?) using (\w[\w\s]+)', RelationType.USES),
        (r'(\w[\w\s]+?) excels? at (\w[\w\s]+)', RelationType.USES),
        (r'(\w[\w\s]+?) involves? (\w[\w\s]+)', RelationType.USES),
        
        # HAS patterns
        (r'(?:types|kinds|forms) of (\w[\w\s]+?)(?::|include|are) (.+)', RelationType.HAS),
        (r'(\w[\w\s]+?) includes? (\w[\w\s]+)', RelationType.HAS),
        
        # EXAMPLE_OF patterns
        (r'(\w[\w\s]+?) (?:is|are) (?:a )?popular (?:algorithm|technique)s?', RelationType.EXAMPLE_OF),
        (r'(?:common |popular )?algorithms? (?:include|are) (.+)', RelationType.EXAMPLE_OF),
    ]
    
    def __init__(self):
        self._nlp = None
        if SPACY_AVAILABLE:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except:
                pass
    
    def _clean_concept(self, text: str) -> str:
        """Clean and normalize a concept name."""
        text = re.sub(r'^(the|a|an)\s+', '', text.strip().lower())
        text = re.sub(r'\s+', ' ', text)
        return text.title()
    
    def _extract_from_patterns(self, text: str) -> List[Relation]:
        """Extract relationships using regex patterns."""
        relations = []
        text_lower = text.lower()
        
        for pattern, rel_type in self.PATTERNS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    source = self._clean_concept(groups[0])
                    target = self._clean_concept(groups[1])
                    
                    if source and target and source != target:
                        relations.append(Relation(
                            source=source,
                            target=target,
                            relation_type=rel_type,
                            confidence=0.8
                        ))
        
        return relations
    
    def _extract_type_hierarchy(self, text: str, keyphrases: List[str]) -> List[Relation]:
        """
        Extract type hierarchies from text.
        E.g., "three types of machine learning: supervised, unsupervised, reinforcement"
        """
        relations = []
        
        # Pattern: "N types/kinds of X: A, B, C"
        pattern = r'(?:three|four|five|several|main|different)?\s*(?:types?|kinds?|forms?)\s+of\s+(\w[\w\s]+?)(?::|;|are|include)\s*(.+)'
        
        for match in re.finditer(pattern, text.lower()):
            parent = self._clean_concept(match.group(1))
            children_text = match.group(2)
            
            # Extract children (comma-separated or "and")
            children = re.split(r',|;|\s+and\s+', children_text)
            
            for child in children:
                child = self._clean_concept(child.strip(' .'))
                if child and len(child) > 2:
                    # Check if child is in keyphrases
                    for kp in keyphrases:
                        if child.lower() in kp.lower() or kp.lower() in child.lower():
                            relations.append(Relation(
                                source=kp,
                                target=parent,
                                relation_type=RelationType.IS_A,
                                confidence=0.9
                            ))
                            break
        
        return relations
    
    def _extract_ml_domain_relations(self, keyphrases: List[str]) -> List[Relation]:
        """
        Extract domain-specific relations for ML/AI topics.
        Uses knowledge about common ML concept relationships.
        """
        relations = []
        
        # Known hierarchies - use PLURAL forms only
        ml_hierarchy = {
            "Machine Learning": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "Deep Learning"],
            "Artificial Intelligence": ["Machine Learning"],
            "Deep Learning": ["Neural Networks"],
            "Supervised Learning": ["Decision Trees", "Random Forests"],
            "Unsupervised Learning": ["Clustering", "Dimensionality Reduction"],
        }
        
        # Known uses
        ml_uses = {
            "Supervised Learning": ["Labeled Data", "Classification", "Regression"],
            "Unsupervised Learning": ["Unlabeled Data"],
            "Reinforcement Learning": ["Robotics", "Game Playing"],
            "Deep Learning": ["Image Recognition", "Natural Language Processing", "Speech Recognition"],
        }
        
        kp_set = {kp.lower() for kp in keyphrases}
        
        # Check hierarchy relationships
        for parent, children in ml_hierarchy.items():
            if parent.lower() in kp_set:
                for child in children:
                    if child.lower() in kp_set:
                        relations.append(Relation(
                            source=child,
                            target=parent,
                            relation_type=RelationType.IS_A,
                            confidence=0.95
                        ))
        
        # Check uses relationships
        for concept, uses in ml_uses.items():
            if concept.lower() in kp_set:
                for use in uses:
                    if use.lower() in kp_set:
                        relations.append(Relation(
                            source=concept,
                            target=use,
                            relation_type=RelationType.USES,
                            confidence=0.85
                        ))
        
        return relations
    
    def extract(self, text: str, keyphrases: List[str]) -> List[Relation]:
        """
        Extract all relationships from text.
        
        Args:
            text: Input text
            keyphrases: List of keyphrase strings
            
        Returns:
            List of Relation objects
        """
        all_relations = []
        
        # 1. Pattern-based extraction
        all_relations.extend(self._extract_from_patterns(text))
        
        # 2. Type hierarchy extraction
        all_relations.extend(self._extract_type_hierarchy(text, keyphrases))
        
        # 3. Domain knowledge (ML/AI)
        all_relations.extend(self._extract_ml_domain_relations(keyphrases))
        
        # Deduplicate
        seen = set()
        unique_relations = []
        for rel in all_relations:
            key = (rel.source.lower(), rel.target.lower(), rel.relation_type)
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)
        
        return unique_relations


# Test
if __name__ == "__main__":
    text = """
    Machine Learning is a subset of Artificial Intelligence. There are three 
    types of machine learning: supervised learning, unsupervised learning, 
    and reinforcement learning. Deep learning uses neural networks.
    """
    
    keyphrases = [
        "Machine Learning", "Artificial Intelligence", "Deep Learning",
        "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning",
        "Neural Networks"
    ]
    
    extractor = RelationExtractor()
    relations = extractor.extract(text, keyphrases)
    
    print("Extracted relations:")
    for rel in relations:
        print(f"  {rel.source} --[{rel.relation_type.value}]--> {rel.target}")
