"""
Comic Layout Module
Assembles generated images into a comic strip layout.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import io
import base64

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .image_generator import GeneratedImage


@dataclass 
class ComicStrip:
    """Represents a completed comic strip."""
    image_data: bytes
    image_base64: str
    panel_count: int
    width: int
    height: int
    
    def save(self, path: Path) -> bool:
        """Save comic strip to file."""
        try:
            with open(path, 'wb') as f:
                f.write(self.image_data)
            return True
        except Exception as e:
            print(f"Error saving comic strip: {e}")
            return False
    
    def to_pil_image(self) -> Optional["Image.Image"]:
        """Convert to PIL Image."""
        if not PIL_AVAILABLE:
            return None
        return Image.open(io.BytesIO(self.image_data))


class ComicLayout:
    """
    Assembles comic panels into a sequential comic strip layout.
    
    Supports various layout configurations:
    - Single row (horizontal strip)
    - Multi-row grid
    - Custom arrangements
    """
    
    def __init__(
        self,
        panel_size: Tuple[int, int] = (400, 400),
        padding: int = 15,
        border_width: int = 3,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        border_color: Tuple[int, int, int] = (0, 0, 0)
    ):
        """
        Initialize the comic layout generator.
        
        Args:
            panel_size: Size of each panel (width, height)
            padding: Padding between panels
            border_width: Width of panel borders
            background_color: RGB background color
            border_color: RGB border color
        """
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required for comic layout generation")
        
        self.panel_size = panel_size
        self.padding = padding
        self.border_width = border_width
        self.background_color = background_color
        self.border_color = border_color
    
    def _add_border(self, image: Image.Image) -> Image.Image:
        """Add border to a panel image."""
        bordered = image.copy()
        draw = ImageDraw.Draw(bordered)
        
        width, height = bordered.size
        
        for i in range(self.border_width):
            draw.rectangle(
                [i, i, width - 1 - i, height - 1 - i],
                outline=self.border_color
            )
        
        return bordered
    
    def _resize_panel(self, image: Image.Image) -> Image.Image:
        """Resize image to panel size."""
        return image.resize(self.panel_size, Image.Resampling.LANCZOS)
    
    def _calculate_dimensions(
        self, 
        panel_count: int, 
        columns: int
    ) -> Tuple[int, int, int]:
        """Calculate strip dimensions based on panel count and columns."""
        rows = (panel_count + columns - 1) // columns
        
        width = (columns * self.panel_size[0] + 
                 (columns + 1) * self.padding)
        height = (rows * self.panel_size[1] + 
                  (rows + 1) * self.padding)
        
        return width, height, rows
    
    def create_strip(
        self,
        images: List[GeneratedImage],
        columns: int = 3,
        title: Optional[str] = None
    ) -> ComicStrip:
        """
        Create a comic strip from generated images.
        
        Args:
            images: List of GeneratedImage objects
            columns: Number of panels per row
            title: Optional title for the comic strip
            
        Returns:
            ComicStrip object with the assembled comic
        """
        # Filter successful images
        valid_images = [img for img in images if img.success and img.image_data]
        
        if not valid_images:
            raise ValueError("No valid images to create comic strip")
        
        panel_count = len(valid_images)
        
        # Calculate dimensions
        width, height, rows = self._calculate_dimensions(panel_count, columns)
        
        # Add space for title if provided
        title_height = 60 if title else 0
        height += title_height
        
        # Create canvas
        canvas = Image.new('RGB', (width, height), self.background_color)
        draw = ImageDraw.Draw(canvas)
        
        # Add title if provided
        if title:
            # Try to use a larger font, fallback to default
            try:
                # Try to load a font (may not work on all systems)
                font = ImageFont.truetype("arial.ttf", 32)
            except:
                font = ImageFont.load_default()
            
            # Center title
            title_bbox = draw.textbbox((0, 0), title, font=font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (width - title_width) // 2
            draw.text((title_x, 15), title, fill=(0, 0, 0), font=font)
        
        # Place panels
        for i, gen_image in enumerate(valid_images):
            row = i // columns
            col = i % columns
            
            # Convert bytes to PIL Image
            panel_img = Image.open(io.BytesIO(gen_image.image_data))
            
            # Resize panel
            panel_img = self._resize_panel(panel_img)
            
            # Add border
            panel_img = self._add_border(panel_img)
            
            # Calculate position
            x = self.padding + col * (self.panel_size[0] + self.padding)
            y = title_height + self.padding + row * (self.panel_size[1] + self.padding)
            
            # Paste panel
            canvas.paste(panel_img, (x, y))
            
            # Add panel number
            panel_num = str(i + 1)
            num_x = x + 10
            num_y = y + 10
            
            # Draw number with background
            draw.ellipse([num_x-5, num_y-5, num_x+20, num_y+20], 
                        fill=(255, 255, 255), outline=(0, 0, 0))
            draw.text((num_x+2, num_y-2), panel_num, fill=(0, 0, 0))
        
        # Convert to bytes
        buffer = io.BytesIO()
        canvas.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        return ComicStrip(
            image_data=image_data,
            image_base64=base64.b64encode(image_data).decode('utf-8'),
            panel_count=panel_count,
            width=width,
            height=height
        )
    
    def create_horizontal_strip(
        self,
        images: List[GeneratedImage],
        title: Optional[str] = None
    ) -> ComicStrip:
        """
        Create a single-row horizontal comic strip.
        
        Args:
            images: List of GeneratedImage objects
            title: Optional title
            
        Returns:
            ComicStrip object
        """
        return self.create_strip(images, columns=len(images), title=title)
    
    def create_vertical_strip(
        self,
        images: List[GeneratedImage],
        title: Optional[str] = None
    ) -> ComicStrip:
        """
        Create a single-column vertical comic strip.
        
        Args:
            images: List of GeneratedImage objects
            title: Optional title
            
        Returns:
            ComicStrip object
        """
        return self.create_strip(images, columns=1, title=title)
    
    def create_grid(
        self,
        images: List[GeneratedImage],
        rows: int = 2,
        title: Optional[str] = None
    ) -> ComicStrip:
        """
        Create a grid layout comic strip.
        
        Args:
            images: List of GeneratedImage objects
            rows: Number of rows
            title: Optional title
            
        Returns:
            ComicStrip object
        """
        panel_count = len([img for img in images if img.success])
        columns = (panel_count + rows - 1) // rows
        return self.create_strip(images, columns=columns, title=title)


# Example usage
if __name__ == "__main__":
    from .image_generator import ImageGenerator
    from .prompt_builder import ImagePrompt
    
    # Generate placeholder images for testing
    generator = ImageGenerator(use_placeholder=True)
    
    prompts = [
        ImagePrompt(
            scene_id=i,
            positive_prompt=f"Scene {i+1} of the story",
            negative_prompt="",
            style_tags=["comic style"]
        )
        for i in range(4)
    ]
    
    images = generator.generate_batch(prompts)
    
    # Create comic strip
    layout = ComicLayout(panel_size=(300, 300))
    comic = layout.create_strip(images, columns=2, title="My Comic Story")
    
    print(f"Comic strip created: {comic.width}x{comic.height}, {comic.panel_count} panels")
    comic.save(Path("test_comic.png"))
    print("Saved to test_comic.png")
