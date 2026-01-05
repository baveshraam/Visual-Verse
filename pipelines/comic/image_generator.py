"""
Image Generator Module - LOCAL GPU VERSION
Generates comic panel images using Stable Diffusion locally on GPU.
"""
import base64
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import io

# Configure logging
logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .prompt_builder import ImagePrompt
import sys
sys.path.append('../..')
from config import SD_DEFAULT_PARAMS


# Global cache for the model pipeline
_CACHED_PIPE = None
_CACHED_MODEL_NAME = None

def get_pipeline(model_name: str = "dreamshaper"):
    """Get or create the Stable Diffusion pipeline (cached globally)."""
    global _CACHED_PIPE, _CACHED_MODEL_NAME
    
    # Return cached model if same model requested
    if _CACHED_PIPE is not None and _CACHED_MODEL_NAME == model_name:
        print("Using cached model pipeline")
        return _CACHED_PIPE
    
    if not DIFFUSERS_AVAILABLE or not TORCH_AVAILABLE:
        print("ERROR: diffusers or torch not available")
        return None
    
    # Model mapping - reliable models
    MODELS = {
        "dreamshaper": "Lykon/dreamshaper-8",  # Best for artistic comic style
        "cartoon": "stablediffusionapi/anything-v5",  # Anime/cartoon style
        "sd15": "runwayml/stable-diffusion-v1-5",  # Classic reliable
        "sdxl": "stabilityai/sdxl-turbo",  # SDXL Turbo for faster generation
    }
    
    model_id = MODELS.get(model_name, MODELS["dreamshaper"])
    
    print(f"=" * 50)
    print(f"Loading model: {model_id}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"=" * 50)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"Loading on device: {device}, dtype: {dtype}")
        
        # Standard SD 1.5 pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        
        # Move to device
        pipe = pipe.to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            pipe.enable_attention_slicing()
            try:
                pipe.enable_vae_slicing()
            except Exception as e:
                logger.debug(f"VAE slicing not available: {e}")
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("xformers enabled")
            except Exception as e:
                logger.debug(f"xformers not available: {e}")
        
        print(f"✓ Model loaded successfully on {device}!")
        
        # Cache it
        _CACHED_PIPE = pipe
        _CACHED_MODEL_NAME = model_name
        
        return pipe
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


@dataclass
class GeneratedImage:
    """Represents a generated image."""
    scene_id: int
    image_data: Optional[bytes]
    image_base64: Optional[str]
    prompt_used: str
    success: bool
    error: Optional[str] = None
    
    def save(self, path: Path) -> bool:
        """Save image to file."""
        if not PIL_AVAILABLE:
            return False
        if self.image_data:
            try:
                img = Image.open(io.BytesIO(self.image_data))
                img.save(path)
                return True
            except Exception as e:
                print(f"Error saving image: {e}")
                return False
        return False
    
    def to_pil_image(self) -> Optional["Image.Image"]:
        """Convert to PIL Image."""
        if not PIL_AVAILABLE or not self.image_data:
            return None
        return Image.open(io.BytesIO(self.image_data))


class ImageGenerator:
    """
    Generates images for comic panels using LOCAL Stable Diffusion on GPU.
    Uses global caching for the model to avoid reloading.
    """
    
    def __init__(
        self,
        model_name: str = "dreamshaper",
        use_placeholder: bool = False,
        **kwargs  # Accept extra args for compatibility
    ):
        self.model_name = model_name
        self.use_placeholder = use_placeholder
        
        # Generation parameters - 512x512 for SD1.5 models
        self.width = 512
        self.height = 512
        self.num_inference_steps = 25
        self.guidance_scale = 7.5
    
    def _get_pipe(self):
        """Get the model pipeline."""
        return get_pipeline(self.model_name)
    
    def _create_placeholder(self, scene_id: int, prompt: str) -> GeneratedImage:
        """Create a placeholder image for testing."""
        if not PIL_AVAILABLE:
            return GeneratedImage(
                scene_id=scene_id,
                image_data=None,
                image_base64=None,
                prompt_used=prompt,
                success=False,
                error="PIL not available"
            )
        
        from PIL import ImageDraw
        
        img = Image.new('RGB', (self.width, self.height), color=(200, 200, 220))
        draw = ImageDraw.Draw(img)
        
        for i in range(3):
            draw.rectangle([i, i, self.width-1-i, self.height-1-i], outline=(100, 100, 120))
        
        draw.text((20, 20), f"Panel {scene_id + 1}", fill=(50, 50, 50))
        draw.text((20, 50), "Prompt:", fill=(80, 80, 80))
        
        y = 70
        words = prompt[:100].split()
        line = ""
        for word in words:
            if len(line + word) > 35:
                draw.text((20, y), line, fill=(60, 60, 60))
                y += 18
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            draw.text((20, y), line, fill=(60, 60, 60))
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        return GeneratedImage(
            scene_id=scene_id,
            image_data=image_data,
            image_base64=base64.b64encode(image_data).decode('utf-8'),
            prompt_used=prompt,
            success=True
        )
    
    def _generate_local(self, prompt: ImagePrompt) -> GeneratedImage:
        """Generate image using local GPU model."""
        pipe = self._get_pipe()
        
        if pipe is None:
            return GeneratedImage(
                scene_id=prompt.scene_id,
                image_data=None,
                image_base64=None,
                prompt_used=prompt.get_full_prompt(),
                success=False,
                error="Model not loaded. Check console for errors."
            )
        
        full_prompt = prompt.get_full_prompt()
        # Use the prompt as-is since prompt_builder handles the styling
        enhanced_prompt = f"{full_prompt}, high quality, detailed, masterpiece"
        
        try:
            print(f"Generating panel {prompt.scene_id + 1}...")
            print(f"  Prompt: {enhanced_prompt[:120]}...")
            
            # Use generator with seed for character consistency
            generator = None
            if prompt.seed is not None:
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
                generator.manual_seed(prompt.seed)
            
            with torch.no_grad():
                result = pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=prompt.negative_prompt,
                    width=self.width,
                    height=self.height,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                )
            
            image = result.images[0]
            
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_data = buffer.getvalue()
            
            print(f"✓ Panel {prompt.scene_id + 1} generated!")
            
            return GeneratedImage(
                scene_id=prompt.scene_id,
                image_data=image_data,
                image_base64=base64.b64encode(image_data).decode('utf-8'),
                prompt_used=full_prompt,
                success=True
            )
            
        except Exception as e:
            print(f"Error generating panel {prompt.scene_id + 1}: {e}")
            import traceback
            traceback.print_exc()
            return GeneratedImage(
                scene_id=prompt.scene_id,
                image_data=None,
                image_base64=None,
                prompt_used=full_prompt,
                success=False,
                error=str(e)
            )
    
    def generate(self, prompt: ImagePrompt) -> GeneratedImage:
        """Generate an image from a prompt."""
        if self.use_placeholder:
            return self._create_placeholder(prompt.scene_id, prompt.get_full_prompt())
        else:
            return self._generate_local(prompt)
    
    def generate_batch(
        self, 
        prompts: List[ImagePrompt],
        delay: float = 0.5
    ) -> List[GeneratedImage]:
        """Generate images for multiple prompts."""
        # Pre-load model before batch
        if not self.use_placeholder:
            print("Pre-loading model for batch generation...")
            pipe = self._get_pipe()
            if pipe is None:
                print("ERROR: Could not load model!")
        
        images = []
        
        for i, prompt in enumerate(prompts):
            print(f"Processing panel {i+1}/{len(prompts)}...")
            image = self.generate(prompt)
            images.append(image)
            
            if image.success:
                print(f"  ✓ Panel {i+1} success")
            else:
                print(f"  ✗ Panel {i+1} failed: {image.error}")
        
        return images
    
    def set_params(self, **kwargs):
        """Set generation parameters."""
        if "width" in kwargs:
            self.width = kwargs["width"]
        if "height" in kwargs:
            self.height = kwargs["height"]
        if "steps" in kwargs:
            self.num_inference_steps = kwargs["steps"]
        if "guidance_scale" in kwargs:
            self.guidance_scale = kwargs["guidance_scale"]
    
    @staticmethod
    def check_gpu() -> dict:
        """Check GPU availability and return info."""
        if not TORCH_AVAILABLE:
            return {"available": False, "error": "PyTorch not installed"}
        
        if torch.cuda.is_available():
            return {
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            }
        else:
            return {"available": False, "error": "CUDA not available"}


# Test if run directly
if __name__ == "__main__":
    print("Testing Image Generator...")
    
    # Check GPU
    gpu_info = ImageGenerator.check_gpu()
    print(f"GPU Info: {gpu_info}")
    
    # Test model loading
    pipe = get_pipeline("dreamshaper")
    if pipe:
        print("Model loaded successfully!")
    else:
        print("Failed to load model")
