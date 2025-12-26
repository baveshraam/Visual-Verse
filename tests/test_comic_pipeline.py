"""
Test suite for the Comic Pipeline modules.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.comic.segmenter import StorySegmenter, StorySegment
from pipelines.comic.extractor import SceneExtractor, SceneDetails
from pipelines.comic.prompt_builder import PromptBuilder, ImagePrompt
from pipelines.comic.image_generator import ImageGenerator, GeneratedImage


class TestStorySegmenter:
    """Test cases for StorySegmenter."""
    
    @pytest.fixture
    def segmenter(self):
        return StorySegmenter(max_segments=4)
    
    @pytest.fixture
    def sample_story(self):
        return """
        Once upon a time, Maya lived in a small village.
        She dreamed of adventure every day.
        
        One morning, she found a hidden path in the forest.
        "This must lead somewhere magical!" she said.
        
        She followed the path and arrived at an ancient tower.
        A mysterious light glowed from the window.
        """
    
    def test_segmenter_initialization(self, segmenter):
        assert segmenter.max_segments == 4
    
    def test_segment_story(self, segmenter, sample_story):
        segments = segmenter.segment(sample_story)
        
        assert isinstance(segments, list)
        assert len(segments) <= 4
        assert all(isinstance(s, StorySegment) for s in segments)
    
    def test_segment_has_required_fields(self, segmenter, sample_story):
        segments = segmenter.segment(sample_story)
        
        if segments:
            seg = segments[0]
            assert hasattr(seg, 'id')
            assert hasattr(seg, 'text')
            assert hasattr(seg, 'sentences')


class TestSceneExtractor:
    """Test cases for SceneExtractor."""
    
    @pytest.fixture
    def extractor(self):
        return SceneExtractor()
    
    def test_extract_scene_details(self, extractor):
        text = """
        Maya, a young girl with bright eyes, walked through the dark forest.
        "I'm not afraid," she whispered.
        """
        
        details = extractor.extract(0, text)
        
        assert isinstance(details, SceneDetails)
        assert details.segment_id == 0
        assert isinstance(details.characters, list)
        assert hasattr(details, 'setting')
        assert hasattr(details, 'mood')
    
    def test_extract_dialogue(self, extractor):
        text = '"Hello there," she said. "How are you?"'
        
        details = extractor.extract(0, text)
        
        assert len(details.dialogue) > 0


class TestPromptBuilder:
    """Test cases for PromptBuilder."""
    
    @pytest.fixture
    def builder(self):
        return PromptBuilder()
    
    def test_build_prompt(self, builder):
        from pipelines.comic.extractor import SceneDetails, Character, Setting, Action
        
        scene = SceneDetails(
            segment_id=0,
            original_text="Test scene",
            characters=[Character(name="Maya", description="young girl")],
            setting=Setting(location="forest", time_of_day="night"),
            actions=[Action(description="walking")],
            dialogue=[],
            mood="mysterious"
        )
        
        prompt = builder.build_prompt(scene)
        
        assert isinstance(prompt, ImagePrompt)
        assert prompt.scene_id == 0
        assert len(prompt.positive_prompt) > 0
    
    def test_set_comic_style(self, builder):
        builder.set_comic_style("manga")
        
        assert "manga" in builder.style_tags[0].lower()


class TestImageGenerator:
    """Test cases for ImageGenerator."""
    
    @pytest.fixture
    def generator(self):
        return ImageGenerator(use_placeholder=True)
    
    def test_generate_placeholder(self, generator):
        prompt = ImagePrompt(
            scene_id=0,
            positive_prompt="test prompt",
            negative_prompt="",
            style_tags=["comic style"]
        )
        
        result = generator.generate(prompt)
        
        assert isinstance(result, GeneratedImage)
        assert result.success == True
        assert result.image_data is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
