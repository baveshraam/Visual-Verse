"""
Graph Builder - HIERARCHICAL VERSION
Builds proper tree-structured graphs from concepts and relationships.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from enum import Enum

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from .keyphrase import Keyphrase
from .relation_extractor import Relation, RelationType


class NodeType(Enum):
    """Node types by hierarchy level."""
    CENTRAL = "central"      # Root node
    PRIMARY = "primary"      # First level children
    SECONDARY = "secondary"  # Second level
    TERTIARY = "tertiary"    # Third level and beyond


@dataclass
class ConceptNode:
    """A concept node in the mind map."""
    id: str
    label: str
    node_type: NodeType
    score: float = 0.0
    parent_id: Optional[str] = None  # For hierarchy tracking
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.node_type.value,
            "score": self.score
        }


@dataclass
class ConceptEdge:
    """An edge between concepts."""
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    label: str = ""
    
    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.relation_type.value
        }


class GraphBuilder:
    """
    Builds hierarchical mind map graphs.
    
    Creates proper tree structure:
    - Central topic at root
    - IS_A relations create parent-child edges
    - USES relations create sibling connections
    """
    
    MAX_NODES = 12
    
    def __init__(self, central_topic: Optional[str] = None):
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required")
        
        self.central_topic = central_topic
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: List[ConceptEdge] = []
        self._node_parents: Dict[str, str] = {}  # child_id -> parent_id
    
    def _normalize_id(self, text: str) -> str:
        """Normalize text to ID."""
        return text.lower().strip().replace(" ", "_").replace("-", "_")
    
    def _clean_label(self, text: str) -> str:
        """Clean label for display."""
        text = text.strip().title()
        if len(text) > 25:
            words = text.split()
            result = ""
            for word in words:
                if len(result) + len(word) + 1 <= 22:
                    result = f"{result} {word}".strip()
                else:
                    break
            text = result + "..." if result else text[:22] + "..."
        return text
    
    def add_node(
        self,
        label: str,
        node_type: NodeType,
        score: float = 0.5,
        parent_id: Optional[str] = None
    ) -> str:
        """Add a node to the graph."""
        node_id = self._normalize_id(label)
        clean_label = self._clean_label(label)
        
        if node_id not in self.nodes and len(self.nodes) < self.MAX_NODES:
            node = ConceptNode(
                id=node_id,
                label=clean_label,
                node_type=node_type,
                score=score,
                parent_id=parent_id
            )
            self.nodes[node_id] = node
            self.graph.add_node(
                node_id,
                label=clean_label,
                type=node_type.value,
                score=score
            )
            
            if parent_id:
                self._node_parents[node_id] = parent_id
        
        return node_id
    
    def add_edge(
        self,
        source_label: str,
        target_label: str,
        relation_type: RelationType,
        weight: float = 1.0
    ) -> bool:
        """Add an edge between nodes."""
        source_id = self._normalize_id(source_label)
        target_id = self._normalize_id(target_label)
        
        # Both nodes must exist
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        
        # No self-loops
        if source_id == target_id:
            return False
        
        # Check for duplicate
        for e in self.edges:
            if e.source_id == source_id and e.target_id == target_id:
                return False
        
        edge = ConceptEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            label=relation_type.value.replace("_", " ")
        )
        self.edges.append(edge)
        
        self.graph.add_edge(source_id, target_id, 
                           type=relation_type.value, weight=weight)
        return True
    
    def _determine_node_type(self, depth: int) -> NodeType:
        """Get node type based on depth in hierarchy."""
        if depth == 0:
            return NodeType.CENTRAL
        elif depth == 1:
            return NodeType.PRIMARY
        elif depth == 2:
            return NodeType.SECONDARY
        else:
            return NodeType.TERTIARY
    
    def build_from_keyphrases_and_relations(
        self,
        keyphrases: List[Keyphrase],
        relations: List[Relation],
        central_topic: Optional[str] = None
    ) -> nx.DiGraph:
        """Build hierarchical graph from keyphrases and relations."""
        
        # Reset
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.edges = []
        self._node_parents = {}
        
        if not keyphrases:
            return self.graph
        
        # Determine central topic
        if central_topic:
            self.central_topic = central_topic
        else:
            self.central_topic = keyphrases[0].phrase
        
        # Build hierarchy from IS_A relations
        # IS_A: child is a type of parent
        # So we want edges from parent -> child
        
        child_to_parent: Dict[str, str] = {}
        for rel in relations:
            if rel.relation_type == RelationType.IS_A:
                # source IS_A target means target is parent of source
                child_to_parent[rel.source.lower()] = rel.target.lower()
        
        # Find root nodes (nodes with no parent or parent is central topic)
        central_lower = self.central_topic.lower()
        
        # Add central node
        self.add_node(self.central_topic, NodeType.CENTRAL, score=1.0)
        
        # Organize keyphrases by depth
        kp_by_phrase = {kp.phrase.lower(): kp for kp in keyphrases}
        
        # First pass: add all nodes
        for kp in keyphrases:
            if kp.phrase.lower() != central_lower:
                parent_phrase = child_to_parent.get(kp.phrase.lower())
                
                if parent_phrase and parent_phrase != central_lower:
                    # Has a non-central parent
                    node_type = NodeType.SECONDARY
                elif parent_phrase == central_lower:
                    # Direct child of central
                    node_type = NodeType.PRIMARY
                else:
                    # No explicit parent - primary by default
                    node_type = NodeType.PRIMARY
                
                self.add_node(kp.phrase, node_type, kp.score)
        
        # Second pass: add edges based on relations
        for rel in relations:
            if rel.relation_type == RelationType.IS_A:
                # child IS_A parent → edge from parent to child
                self.add_edge(rel.target, rel.source, RelationType.PART_OF, rel.confidence)
            elif rel.relation_type == RelationType.USES:
                self.add_edge(rel.source, rel.target, RelationType.USES, rel.confidence)
            else:
                self.add_edge(rel.source, rel.target, rel.relation_type, rel.confidence)
        
        # Connect orphan nodes to central topic
        central_id = self._normalize_id(self.central_topic)
        for node_id in self.nodes:
            if node_id != central_id:
                # Check if has any incoming or outgoing edges
                has_edges = any(
                    e.source_id == node_id or e.target_id == node_id
                    for e in self.edges
                )
                if not has_edges:
                    self.add_edge(self.central_topic, self.nodes[node_id].label, 
                                 RelationType.RELATED_TO, 0.5)
        
        return self.graph
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "central_topic": self.central_topic,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "stats": {"nodes": len(self.nodes), "edges": len(self.edges)}
        }


# Test
if __name__ == "__main__":
    from .keyphrase import Keyphrase
    from .relation_extractor import Relation
    
    keyphrases = [
        Keyphrase("Machine Learning", 0.95, "compound"),
        Keyphrase("Supervised Learning", 0.9, "compound"),
        Keyphrase("Deep Learning", 0.85, "compound"),
        Keyphrase("Neural Networks", 0.8, "compound"),
    ]
    
    relations = [
        Relation("Supervised Learning", "Machine Learning", RelationType.IS_A, 0.95),
        Relation("Deep Learning", "Machine Learning", RelationType.IS_A, 0.9),
        Relation("Deep Learning", "Neural Networks", RelationType.USES, 0.85),
    ]
    
    builder = GraphBuilder()
    graph = builder.build_from_keyphrases_and_relations(keyphrases, relations)
    
    print(f"Built graph: {len(builder.nodes)} nodes, {len(builder.edges)} edges")
    for e in builder.edges:
        print(f"  {e.source_id} --[{e.relation_type.value}]--> {e.target_id}")
