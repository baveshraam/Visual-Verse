"""
Image Processing Utilities
Handles image manipulation for comic strip generation.
"""
import base64
import io
from pathlib import Path
from typing import List, Optional, Tuple, Union

try:
    from PIL import Image
except ImportError:
    Image = None


class ImageUtils:
    """Utility class for image processing operations."""
    
    @staticmethod
    def check_pil_available() -> bool:
        """Check if PIL/Pillow is available."""
        return Image is not None
    
    @staticmethod
    def load_image(path: Union[str, Path]) -> Optional["Image.Image"]:
        """
        Load an image from file path.
        
        Args:
            path: Path to image file
            
        Returns:
            PIL Image object or None if failed
        """
        if not ImageUtils.check_pil_available():
            raise ImportError("Pillow is required for image processing")
        
        try:
            return Image.open(path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    @staticmethod
    def save_image(image: "Image.Image", path: Union[str, Path], format: str = "PNG") -> bool:
        """
        Save an image to file.
        
        Args:
            image: PIL Image object
            path: Output file path
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image.save(path, format=format)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    @staticmethod
    def resize_image(
        image: "Image.Image",
        size: Tuple[int, int],
        maintain_aspect: bool = True
    ) -> "Image.Image":
        """
        Resize an image.
        
        Args:
            image: PIL Image object
            size: Target size (width, height)
            maintain_aspect: If True, maintains aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect:
            image.thumbnail(size, Image.Resampling.LANCZOS)
            return image
        else:
            return image.resize(size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def create_comic_strip(
        panels: List["Image.Image"],
        panel_size: Tuple[int, int] = (512, 512),
        columns: int = 3,
        padding: int = 10,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> "Image.Image":
        """
        Create a comic strip layout from individual panel images.
        
        Args:
            panels: List of PIL Image objects for each panel
            panel_size: Size of each panel (width, height)
            columns: Number of panels per row
            padding: Padding between panels in pixels
            background_color: RGB background color
            
        Returns:
            Combined comic strip image
        """
        if not ImageUtils.check_pil_available():
            raise ImportError("Pillow is required for image processing")
        
        if not panels:
            raise ValueError("At least one panel is required")
        
        # Calculate dimensions
        num_panels = len(panels)
        rows = (num_panels + columns - 1) // columns  # Ceiling division
        
        strip_width = columns * panel_size[0] + (columns + 1) * padding
        strip_height = rows * panel_size[1] + (rows + 1) * padding
        
        # Create the comic strip canvas
        strip = Image.new('RGB', (strip_width, strip_height), background_color)
        
        # Place each panel
        for i, panel in enumerate(panels):
            row = i // columns
            col = i % columns
            
            # Resize panel if necessary
            resized_panel = ImageUtils.resize_image(
                panel.copy(), 
                panel_size, 
                maintain_aspect=False
            )
            
            # Calculate position
            x = padding + col * (panel_size[0] + padding)
            y = padding + row * (panel_size[1] + padding)
            
            strip.paste(resized_panel, (x, y))
        
        return strip
    
    @staticmethod
    def image_to_base64(image: "Image.Image", format: str = "PNG") -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            format: Image format
            
        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    @staticmethod
    def base64_to_image(base64_string: str) -> "Image.Image":
        """
        Convert base64 string to PIL Image.
        
        Args:
            base64_string: Base64 encoded image data
            
        Returns:
            PIL Image object
        """
        if not ImageUtils.check_pil_available():
            raise ImportError("Pillow is required for image processing")
        
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    
    @staticmethod
    def add_panel_border(
        image: "Image.Image",
        border_width: int = 3,
        border_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> "Image.Image":
        """
        Add a border around an image panel.
        
        Args:
            image: PIL Image object
            border_width: Width of border in pixels
            border_color: RGB color of border
            
        Returns:
            Image with border added
        """
        if not ImageUtils.check_pil_available():
            raise ImportError("Pillow is required for image processing")
        
        from PIL import ImageDraw
        
        bordered = image.copy()
        draw = ImageDraw.Draw(bordered)
        
        width, height = bordered.size
        
        # Draw border lines
        for i in range(border_width):
            draw.rectangle(
                [i, i, width - 1 - i, height - 1 - i],
                outline=border_color
            )
        
        return bordered
    
    @staticmethod
    def create_placeholder_image(
        size: Tuple[int, int] = (512, 512),
        text: str = "Panel",
        background_color: Tuple[int, int, int] = (200, 200, 200)
    ) -> "Image.Image":
        """
        Create a placeholder image with text (for testing/development).
        
        Args:
            size: Image size (width, height)
            text: Text to display on placeholder
            background_color: RGB background color
            
        Returns:
            Placeholder image
        """
        if not ImageUtils.check_pil_available():
            raise ImportError("Pillow is required for image processing")
        
        from PIL import ImageDraw
        
        image = Image.new('RGB', size, background_color)
        draw = ImageDraw.Draw(image)
        
        # Draw text in center
        text_bbox = draw.textbbox((0, 0), text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        draw.text((x, y), text, fill=(100, 100, 100))
        
        return image
