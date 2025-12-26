"""
FastAPI Routes for VisualVerse
REST API endpoints for text classification and visualization generation.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import base64

# Import core modules
from core.classifier import TextClassifier, TextType
from core.router import PipelineRouter

# Import pipeline modules
from pipelines.comic.segmenter import StorySegmenter
from pipelines.comic.extractor import SceneExtractor
from pipelines.comic.prompt_builder import PromptBuilder
from pipelines.comic.image_generator import ImageGenerator
from pipelines.comic.layout import ComicLayout

from pipelines.mindmap.keyphrase import KeyphraseExtractor
from pipelines.mindmap.relation_extractor import RelationExtractor
from pipelines.mindmap.graph_builder import GraphBuilder
from pipelines.mindmap.visualizer import MindMapVisualizer


# Initialize FastAPI app
app = FastAPI(
    title="VisualVerse API",
    description="Dual-mode NLP system for converting text to Comics or Mind-Maps",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (lazy loading)
_classifier = None
_router = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = TextClassifier(use_semantic=False)
    return _classifier


def get_router():
    global _router
    if _router is None:
        _router = PipelineRouter(get_classifier())
    return _router


# Request/Response Models
class TextInput(BaseModel):
    """Input model for text processing."""
    text: str = Field(..., min_length=10, description="Input text to process")
    force_mode: Optional[str] = Field(None, description="Force 'comic' or 'mindmap' mode")


class ClassificationResponse(BaseModel):
    """Response model for text classification."""
    text_type: str
    confidence: float
    narrative_score: float
    informational_score: float
    suggested_pipeline: str


class ComicRequest(BaseModel):
    """Request model for comic generation."""
    text: str
    style: str = "western"
    max_panels: int = 6
    use_placeholder: bool = True


class MindMapRequest(BaseModel):
    """Request model for mind map generation."""
    text: str
    max_keywords: int = 15
    central_topic: Optional[str] = None
    theme: str = "light"


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "VisualVerse API",
        "version": "1.0.0",
        "endpoints": {
            "classify": "/api/classify",
            "process": "/api/process",
            "comic": "/api/comic",
            "mindmap": "/api/mindmap"
        }
    }


@app.post("/api/classify", response_model=ClassificationResponse)
async def classify_text(input_data: TextInput):
    """
    Classify text as Narrative or Informational.
    
    Returns classification result with confidence scores.
    """
    classifier = get_classifier()
    result = classifier.classify(input_data.text)
    
    return ClassificationResponse(
        text_type=result.text_type.value,
        confidence=result.confidence,
        narrative_score=result.narrative_score,
        informational_score=result.informational_score,
        suggested_pipeline="comic" if result.text_type == TextType.NARRATIVE else "mindmap"
    )


@app.post("/api/process")
async def process_text(input_data: TextInput):
    """
    Process text through the appropriate pipeline.
    
    Automatically classifies and routes to Comic or Mind-Map pipeline.
    """
    router = get_router()
    route_result = router.route(input_data.text)
    
    # Override if force_mode specified
    pipeline = input_data.force_mode if input_data.force_mode else route_result.pipeline
    
    response = {
        "classification": route_result.classification.to_dict(),
        "pipeline": pipeline,
        "result": None,
        "error": None
    }
    
    try:
        if pipeline == "comic":
            # Run comic pipeline
            result = await generate_comic_internal(input_data.text)
            response["result"] = result
        else:
            # Run mind-map pipeline
            result = await generate_mindmap_internal(input_data.text)
            response["result"] = result
            
    except Exception as e:
        response["error"] = str(e)
    
    return response


async def generate_comic_internal(
    text: str,
    style: str = "western",
    max_panels: int = 6,
    use_placeholder: bool = True
) -> Dict[str, Any]:
    """Internal function to generate comic from text."""
    # Segment story
    segmenter = StorySegmenter(max_segments=max_panels)
    segments = segmenter.segment(text)
    
    # Extract scene details
    extractor = SceneExtractor()
    scenes = [extractor.extract(seg.id, seg.text) for seg in segments]
    
    # Build prompts
    prompt_builder = PromptBuilder()
    prompt_builder.set_comic_style(style)
    prompts = prompt_builder.build_prompts(scenes)
    
    # Generate images
    generator = ImageGenerator(use_placeholder=use_placeholder)
    images = generator.generate_batch(prompts)
    
    # Create comic strip
    layout = ComicLayout()
    comic = layout.create_strip(images, columns=min(3, len(images)), title="Generated Comic")
    
    return {
        "segments": [seg.to_dict() for seg in segments],
        "scenes": [scene.to_dict() for scene in scenes],
        "prompts": [p.to_dict() for p in prompts],
        "comic_image_base64": comic.image_base64,
        "panel_count": comic.panel_count,
        "dimensions": {"width": comic.width, "height": comic.height}
    }


async def generate_mindmap_internal(
    text: str,
    max_keywords: int = 15,
    central_topic: Optional[str] = None,
    theme: str = "light"
) -> Dict[str, Any]:
    """Internal function to generate mind map from text."""
    # Extract keyphrases
    keyphrase_extractor = KeyphraseExtractor(max_keywords=max_keywords)
    keyphrases = keyphrase_extractor.extract(text)
    
    # Extract relations
    relation_extractor = RelationExtractor()
    keyphrase_strings = [kp.phrase for kp in keyphrases]
    relations = relation_extractor.extract(text, keyphrase_strings)
    
    # Build graph
    graph_builder = GraphBuilder(central_topic=central_topic)
    graph = graph_builder.build_from_keyphrases_and_relations(keyphrases, relations)
    
    # Visualize
    visualizer = MindMapVisualizer()
    visualizer.set_theme(theme)
    network = visualizer.visualize(graph_builder, "Mind Map")
    html_content = visualizer.get_html_string()
    
    return {
        "keyphrases": [kp.to_dict() for kp in keyphrases],
        "relations": [rel.to_dict() for rel in relations],
        "graph": graph_builder.to_dict(),
        "hierarchy": graph_builder.get_hierarchy(),
        "visualization_html": html_content
    }


@app.post("/api/comic")
async def generate_comic(request: ComicRequest):
    """
    Generate a comic strip from narrative text.
    
    Returns comic strip image and scene details.
    """
    try:
        result = await generate_comic_internal(
            request.text,
            request.style,
            request.max_panels,
            request.use_placeholder
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mindmap")
async def generate_mindmap(request: MindMapRequest):
    """
    Generate a mind map from informational text.
    
    Returns interactive mind map visualization.
    """
    try:
        result = await generate_mindmap_internal(
            request.text,
            request.max_keywords,
            request.central_topic,
            request.theme
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mindmap/view/{map_id}", response_class=HTMLResponse)
async def view_mindmap(map_id: str):
    """
    View a generated mind map (placeholder for session-based storage).
    """
    # In a full implementation, this would retrieve stored mind maps
    return HTMLResponse(content="<html><body>Mind map viewer placeholder</body></html>")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


# Run with: uvicorn api.routes:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
