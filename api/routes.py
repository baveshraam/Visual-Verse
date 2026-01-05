"""
FastAPI Routes for VisualVerse
REST API endpoints for text classification and visualization generation.
Supports all IndicTrans2 languages (22 Indic languages + English).
"""
import sys
import io
import time

# Fix Windows console encoding for Unicode text
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import base64
import logging

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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

# Import multilingual processor and file logging
from utils.tamil_processor import get_tamil_processor, TamilProcessor
from utils.logging_config import log_api_call, log_api_response, log_pipeline_step, log_error


# Initialize FastAPI app
app = FastAPI(
    title="VisualVerse API",
    description="Dual-mode NLP system for converting text to Comics or Mind-Maps. Now with Tamil language support!",
    version="1.1.0"
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
_tamil_processor = None


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


def get_processor() -> TamilProcessor:
    """Get or initialize Tamil processor."""
    global _tamil_processor
    if _tamil_processor is None:
        _tamil_processor = get_tamil_processor()
    return _tamil_processor


# Request/Response Models
class TextInput(BaseModel):
    """Input model for text processing."""
    text: str = Field(..., min_length=10, description="Input text to process (Tamil or English)")
    force_mode: Optional[str] = Field(None, description="Force 'comic' or 'mindmap' mode")


class TranslationInfo(BaseModel):
    """Translation information for response."""
    original_language: str
    was_translated: bool
    translated_text: Optional[str] = None
    translation_warning: Optional[str] = None


class ClassificationResponse(BaseModel):
    """Response model for text classification."""
    text_type: str
    confidence: float
    narrative_score: float
    informational_score: float
    suggested_pipeline: str
    translation_info: Optional[TranslationInfo] = None


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
    Supports all IndicTrans2 languages - automatically detects and translates.
    
    Returns classification result with confidence scores.
    """
    start_time = time.time()
    log_api_call("/api/classify", "POST", len(input_data.text))
    log_pipeline_step("CLASSIFY", "START", f"Text length: {len(input_data.text)} chars")
    
    # Get multilingual processor
    processor = get_processor()
    
    # Process text (detect language and translate if needed)
    log_pipeline_step("CLASSIFY", "LANGUAGE_DETECT", "Detecting language...")
    translation_result = processor.process_text(input_data.text)
    text_to_classify = translation_result.translated_text
    log_pipeline_step("CLASSIFY", "LANGUAGE_RESULT", f"Detected: {translation_result.source_language}, Translated: {translation_result.was_translated}")
    
    # Build translation info
    translation_info = TranslationInfo(
        original_language=translation_result.source_language,
        was_translated=translation_result.was_translated,
        translated_text=translation_result.translated_text if translation_result.was_translated else None,
        translation_warning=translation_result.error_message
    )
    
    if translation_result.was_translated:
        logger.info(f"Classified translated text from {translation_result.source_language}")
        log_pipeline_step("CLASSIFY", "TRANSLATED", f"From {translation_result.source_language} to English")
    
    # Classify the (possibly translated) text
    log_pipeline_step("CLASSIFY", "CLASSIFYING", "Running text classifier...")
    classifier = get_classifier()
    result = classifier.classify(text_to_classify)
    log_pipeline_step("CLASSIFY", "RESULT", f"Type: {result.text_type.value}, Confidence: {result.confidence:.2%}")
    
    duration_ms = (time.time() - start_time) * 1000
    log_api_response("/api/classify", 200, duration_ms)
    
    return ClassificationResponse(
        text_type=result.text_type.value,
        confidence=result.confidence,
        narrative_score=result.narrative_score,
        informational_score=result.informational_score,
        suggested_pipeline="comic" if result.text_type == TextType.NARRATIVE else "mindmap",
        translation_info=translation_info
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
    use_placeholder: bool = True,
    is_tamil_origin: bool = False
) -> Dict[str, Any]:
    """
    Internal function to generate comic from text.
    
    Args:
        text: The text to generate comic from (should be in English)
        style: Art style for the comic
        max_panels: Maximum number of panels
        use_placeholder: Whether to use placeholder images
        is_tamil_origin: Whether the original text was in Indic language (for cultural context)
    """
    log_pipeline_step("COMIC", "START", f"Style: {style}, Max panels: {max_panels}, Placeholder: {use_placeholder}")
    
    # Step 1: Segment story
    log_pipeline_step("COMIC", "SEGMENT", "Segmenting story into panels...")
    segmenter = StorySegmenter(max_segments=max_panels)
    segments = segmenter.segment(text)
    log_pipeline_step("COMIC", "SEGMENT_DONE", f"Created {len(segments)} segments")
    
    # Step 2: Extract scene details
    log_pipeline_step("COMIC", "EXTRACT", "Extracting scene details (characters, settings, mood)...")
    extractor = SceneExtractor()
    scenes = [extractor.extract(seg.id, seg.text) for seg in segments]
    log_pipeline_step("COMIC", "EXTRACT_DONE", f"Extracted {len(scenes)} scenes")
    
    # Step 3: Build prompts
    log_pipeline_step("COMIC", "PROMPTS", "Building image generation prompts...")
    prompt_builder = PromptBuilder()
    prompt_builder.set_comic_style(style)
    prompts = prompt_builder.build_prompts(scenes)
    log_pipeline_step("COMIC", "PROMPTS_DONE", f"Built {len(prompts)} prompts")
    
    # If Indic origin, inject cultural context into prompts
    if is_tamil_origin:
        processor = get_processor()
        cultural_context = processor.get_cultural_context("ta")
        for prompt in prompts:
            prompt.positive_prompt = f"{prompt.positive_prompt}, {cultural_context}"
        log_pipeline_step("COMIC", "CULTURAL", f"Injected cultural context into {len(prompts)} prompts")
    
    # Step 4: Generate images
    log_pipeline_step("COMIC", "GENERATE", "Generating images (placeholder mode)..." if use_placeholder else "Generating images with AI...")
    generator = ImageGenerator(use_placeholder=use_placeholder)
    images = generator.generate_batch(prompts)
    log_pipeline_step("COMIC", "GENERATE_DONE", f"Generated {len(images)} images")
    
    # Step 5: Create comic strip
    log_pipeline_step("COMIC", "LAYOUT", "Creating comic strip layout...")
    layout = ComicLayout()
    comic = layout.create_strip(images, columns=min(3, len(images)), title="Generated Comic")
    log_pipeline_step("COMIC", "COMPLETE", f"Comic created: {comic.panel_count} panels, {comic.width}x{comic.height}px")
    
    return {
        "segments": [seg.to_dict() for seg in segments],
        "scenes": [scene.to_dict() for scene in scenes],
        "prompts": [p.to_dict() for p in prompts],
        "comic_image_base64": comic.image_base64,
        "panel_count": comic.panel_count,
        "dimensions": {"width": comic.width, "height": comic.height},
        "is_tamil_origin": is_tamil_origin
    }


async def generate_mindmap_internal(
    text: str,
    max_keywords: int = 15,
    central_topic: Optional[str] = None,
    theme: str = "light"
) -> Dict[str, Any]:
    """Internal function to generate mind map from text."""
    log_pipeline_step("MINDMAP", "START", f"Max keywords: {max_keywords}, Theme: {theme}")
    
    # Step 1: Extract keyphrases
    log_pipeline_step("MINDMAP", "KEYPHRASE", "Extracting keyphrases...")
    keyphrase_extractor = KeyphraseExtractor(max_keywords=max_keywords)
    keyphrases = keyphrase_extractor.extract(text)
    log_pipeline_step("MINDMAP", "KEYPHRASE_DONE", f"Extracted {len(keyphrases)} keyphrases")
    
    # Step 2: Extract relations
    log_pipeline_step("MINDMAP", "RELATIONS", "Finding relationships between concepts...")
    relation_extractor = RelationExtractor()
    keyphrase_strings = [kp.phrase for kp in keyphrases]
    relations = relation_extractor.extract(text, keyphrase_strings)
    log_pipeline_step("MINDMAP", "RELATIONS_DONE", f"Found {len(relations)} relations")
    
    # Step 3: Build graph
    log_pipeline_step("MINDMAP", "GRAPH", "Building knowledge graph...")
    graph_builder = GraphBuilder(central_topic=central_topic)
    graph = graph_builder.build_from_keyphrases_and_relations(keyphrases, relations)
    log_pipeline_step("MINDMAP", "GRAPH_DONE", f"Graph: {len(graph_builder.nodes)} nodes, {len(graph_builder.edges)} edges")
    
    # Step 4: Visualize
    log_pipeline_step("MINDMAP", "VISUALIZE", "Creating visualization...")
    visualizer = MindMapVisualizer()
    visualizer.set_theme(theme)
    network = visualizer.visualize(graph_builder, "Mind Map")
    html_content = visualizer.get_html_string()
    log_pipeline_step("MINDMAP", "COMPLETE", f"Visualization created with {len(html_content)} bytes of HTML")
    
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
    Supports Tamil input - automatically detects, translates, and adds cultural context.
    
    Returns comic strip image and scene details.
    """
    try:
        # Process Tamil if detected
        processor = get_processor()
        translation_result = processor.process_text(request.text)
        text_to_process = translation_result.translated_text
        is_tamil = translation_result.was_translated
        
        if is_tamil:
            logger.info(f"Processing Tamil comic request - translated text: {text_to_process[:100]}...")
        
        result = await generate_comic_internal(
            text_to_process,
            request.style,
            request.max_panels,
            request.use_placeholder,
            is_tamil_origin=is_tamil
        )
        
        # Add translation info to response
        result["translation_info"] = {
            "original_language": translation_result.source_language,
            "was_translated": translation_result.was_translated,
            "translation_warning": translation_result.error_message
        }
        
        return result
    except Exception as e:
        logger.error(f"Comic generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mindmap")
async def generate_mindmap(request: MindMapRequest):
    """
    Generate a mind map from informational text.
    Supports Tamil input - automatically detects and translates.
    
    Returns interactive mind map visualization.
    """
    try:
        # Process Tamil if detected
        processor = get_processor()
        translation_result = processor.process_text(request.text)
        text_to_process = translation_result.translated_text
        
        if translation_result.was_translated:
            logger.info(f"Processing Tamil mindmap request - translated text: {text_to_process[:100]}...")
        
        result = await generate_mindmap_internal(
            text_to_process,
            request.max_keywords,
            request.central_topic,
            request.theme
        )
        
        # Add translation info to response
        result["translation_info"] = {
            "original_language": translation_result.source_language,
            "was_translated": translation_result.was_translated,
            "translation_warning": translation_result.error_message
        }
        
        return result
    except Exception as e:
        logger.error(f"Mindmap generation failed: {e}")
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
    """Health check endpoint with GPU status."""
    import torch
    
    gpu_info = {
        "available": False,
        "device": None,
        "memory": None
    }
    
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        }
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "gpu": gpu_info
    }


@app.get("/api/gpu")
async def get_gpu_status():
    """Get GPU status and availability."""
    import torch
    
    if torch.cuda.is_available():
        return {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            "cuda_version": torch.version.cuda or "N/A"
        }
    else:
        return {
            "available": False,
            "error": "CUDA not available. Running on CPU.",
            "torch_version": torch.__version__,
            "torch_location": torch.__file__
        }


# Run with: uvicorn api.routes:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
