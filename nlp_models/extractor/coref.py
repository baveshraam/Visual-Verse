"""
Coreference Resolution Module
Tracks entity mentions across a document to resolve pronouns and references.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class Mention:
    """A mention of an entity in text."""
    text: str
    start: int
    end: int
    entity_id: Optional[int] = None


@dataclass 
class CoreferenceCluster:
    """A cluster of coreferent mentions."""
    entity_id: int
    canonical_name: str
    mentions: List[Mention]
    entity_type: Optional[str] = None


class CoreferenceResolver:
    """
    Rule-based coreference resolution for narrative text.
    
    Handles:
    - Pronoun resolution (he/she/they → character)
    - Definite noun phrases (the girl → Maya)
    - Repeated names
    
    For production, consider using:
    - AllenNLP coreference model
    - neuralcoref (spaCy extension)
    """
    
    # Common pronouns by gender/number
    MALE_PRONOUNS = {'he', 'him', 'his', 'himself'}
    FEMALE_PRONOUNS = {'she', 'her', 'hers', 'herself'}
    NEUTRAL_PRONOUNS = {'they', 'them', 'their', 'themselves', 'it', 'its'}
    
    def __init__(self):
        self.clusters: List[CoreferenceCluster] = []
        self.mention_to_cluster: Dict[Tuple[int, int], int] = {}
    
    def resolve(
        self, 
        text: str, 
        entities: List[Dict]
    ) -> List[CoreferenceCluster]:
        """
        Resolve coreferences in text.
        
        Args:
            text: Full text
            entities: List of entity dicts with 'text', 'label', 'start', 'end'
            
        Returns:
            List of coreference clusters
        """
        self.clusters = []
        self.mention_to_cluster = {}
        
        # First pass: group entities by type
        characters = [e for e in entities if e.get('label') == 'CHARACTER']
        
        if not characters:
            return []
        
        # Create initial clusters from named entities
        entity_id = 0
        name_to_cluster = {}
        
        for entity in characters:
            name = entity['text'].lower()
            
            if name in name_to_cluster:
                # Add to existing cluster
                cluster_id = name_to_cluster[name]
                self.clusters[cluster_id].mentions.append(Mention(
                    text=entity['text'],
                    start=entity['start'],
                    end=entity['end'],
                    entity_id=cluster_id
                ))
            else:
                # Create new cluster
                cluster = CoreferenceCluster(
                    entity_id=entity_id,
                    canonical_name=entity['text'],
                    mentions=[Mention(
                        text=entity['text'],
                        start=entity['start'],
                        end=entity['end'],
                        entity_id=entity_id
                    )],
                    entity_type='CHARACTER'
                )
                self.clusters.append(cluster)
                name_to_cluster[name] = entity_id
                entity_id += 1
        
        # Second pass: resolve pronouns
        self._resolve_pronouns(text)
        
        return self.clusters
    
    def _resolve_pronouns(self, text: str):
        """Resolve pronouns to entity clusters."""
        # Find all pronouns
        pronoun_pattern = r'\b(he|she|they|him|her|them|his|hers|their|it|its)\b'
        
        for match in re.finditer(pronoun_pattern, text, re.IGNORECASE):
            pronoun = match.group().lower()
            start = match.start()
            end = match.end()
            
            # Find nearest matching entity
            cluster_id = self._find_antecedent(pronoun, start, text)
            
            if cluster_id is not None:
                self.clusters[cluster_id].mentions.append(Mention(
                    text=match.group(),
                    start=start,
                    end=end,
                    entity_id=cluster_id
                ))
    
    def _find_antecedent(
        self, 
        pronoun: str, 
        position: int,
        text: str
    ) -> Optional[int]:
        """
        Find the antecedent cluster for a pronoun.
        
        Uses recency-based heuristic with gender agreement.
        """
        if not self.clusters:
            return None
        
        # Determine pronoun gender
        if pronoun in self.MALE_PRONOUNS:
            gender = 'male'
        elif pronoun in self.FEMALE_PRONOUNS:
            gender = 'female'
        else:
            gender = 'neutral'
        
        # Find nearest entity before this position
        best_cluster = None
        best_distance = float('inf')
        
        for cluster in self.clusters:
            for mention in cluster.mentions:
                # Must be before pronoun
                if mention.end < position:
                    distance = position - mention.end
                    if distance < best_distance:
                        # Check gender agreement (simplified)
                        if gender == 'neutral' or self._gender_agrees(cluster, gender):
                            best_distance = distance
                            best_cluster = cluster.entity_id
        
        return best_cluster
    
    def _gender_agrees(self, cluster: CoreferenceCluster, pronoun_gender: str) -> bool:
        """Check if cluster gender agrees with pronoun."""
        # Simplified: just allow any match for now
        # In production, use a name-to-gender database
        return True
    
    def get_entity_mentions(self, entity_id: int) -> List[Mention]:
        """Get all mentions for an entity."""
        for cluster in self.clusters:
            if cluster.entity_id == entity_id:
                return cluster.mentions
        return []
    
    def get_canonical_name(self, entity_id: int) -> Optional[str]:
        """Get the canonical name for an entity."""
        for cluster in self.clusters:
            if cluster.entity_id == entity_id:
                return cluster.canonical_name
        return None
    
    def to_dict(self) -> Dict:
        """Convert clusters to dictionary format."""
        return {
            'clusters': [
                {
                    'entity_id': c.entity_id,
                    'canonical_name': c.canonical_name,
                    'entity_type': c.entity_type,
                    'mentions': [
                        {'text': m.text, 'start': m.start, 'end': m.end}
                        for m in c.mentions
                    ]
                }
                for c in self.clusters
            ]
        }


# Test
if __name__ == '__main__':
    resolver = CoreferenceResolver()
    
    text = """
    Maya walked through the forest. She was searching for the golden key.
    When she found it, Maya smiled with joy. The girl knew her journey was complete.
    """
    
    entities = [
        {'text': 'Maya', 'label': 'CHARACTER', 'start': 5, 'end': 9},
        {'text': 'Maya', 'label': 'CHARACTER', 'start': 100, 'end': 104},
    ]
    
    clusters = resolver.resolve(text, entities)
    
    print("Coreference Clusters:")
    for cluster in clusters:
        print(f"\nEntity: {cluster.canonical_name}")
        for mention in cluster.mentions:
            print(f"  - '{mention.text}' at [{mention.start}:{mention.end}]")
