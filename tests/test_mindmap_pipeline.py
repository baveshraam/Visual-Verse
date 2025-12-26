"""
Test suite for the Mind-Map Pipeline modules.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.mindmap.keyphrase import KeyphraseExtractor, Keyphrase
from pipelines.mindmap.relation_extractor import RelationExtractor, Relation, RelationType
from pipelines.mindmap.graph_builder import GraphBuilder, NodeType


class TestKeyphraseExtractor:
    """Test cases for KeyphraseExtractor."""
    
    @pytest.fixture
    def extractor(self):
        return KeyphraseExtractor(max_keywords=10)
    
    @pytest.fixture
    def sample_text(self):
        return """
        Machine learning is a subset of artificial intelligence that enables 
        computers to learn from data. Deep learning uses neural networks 
        with multiple layers. Natural language processing is an application 
        of machine learning.
        """
    
    def test_extractor_initialization(self, extractor):
        assert extractor.max_keywords == 10
    
    def test_extract_keyphrases(self, extractor, sample_text):
        keyphrases = extractor.extract(sample_text)
        
        assert isinstance(keyphrases, list)
        assert len(keyphrases) > 0
        assert all(isinstance(kp, Keyphrase) for kp in keyphrases)
    
    def test_keyphrase_has_score(self, extractor, sample_text):
        keyphrases = extractor.extract(sample_text)
        
        if keyphrases:
            kp = keyphrases[0]
            assert hasattr(kp, 'score')
            assert 0 <= kp.score <= 1
    
    def test_extract_topics(self, extractor, sample_text):
        topics = extractor.extract_topics(sample_text, n_topics=5)
        
        assert isinstance(topics, list)
        assert len(topics) <= 5


class TestRelationExtractor:
    """Test cases for RelationExtractor."""
    
    @pytest.fixture
    def extractor(self):
        return RelationExtractor()
    
    def test_extract_relations(self, extractor):
        text = """
        Machine learning is a subset of artificial intelligence.
        Deep learning uses neural networks.
        """
        keyphrases = ["machine learning", "artificial intelligence", 
                      "deep learning", "neural networks"]
        
        relations = extractor.extract(text, keyphrases)
        
        assert isinstance(relations, list)
        assert all(isinstance(r, Relation) for r in relations)
    
    def test_relation_has_type(self, extractor):
        text = "Machine learning is a subset of artificial intelligence."
        keyphrases = ["machine learning", "artificial intelligence"]
        
        relations = extractor.extract(text, keyphrases)
        
        if relations:
            assert hasattr(relations[0], 'relation_type')
            assert isinstance(relations[0].relation_type, RelationType)


class TestGraphBuilder:
    """Test cases for GraphBuilder."""
    
    @pytest.fixture
    def builder(self):
        return GraphBuilder()
    
    def test_build_graph(self, builder):
        keyphrases = [
            Keyphrase("machine learning", 0.9, "test"),
            Keyphrase("deep learning", 0.8, "test"),
        ]
        relations = [
            Relation("deep learning", "machine learning", 
                    RelationType.PART_OF, 0.8)
        ]
        
        graph = builder.build_from_keyphrases_and_relations(keyphrases, relations)
        
        assert graph is not None
        assert len(builder.nodes) >= 2
    
    def test_graph_to_dict(self, builder):
        keyphrases = [Keyphrase("test topic", 0.9, "test")]
        
        builder.build_from_keyphrases_and_relations(keyphrases, [])
        result = builder.to_dict()
        
        assert 'nodes' in result
        assert 'edges' in result
        assert 'central_topic' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
