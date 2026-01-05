"""
Multilingual Processor Module
Handles language detection and translation for ALL IndicTrans2 supported languages.

Supported Languages (22 Indic + English):
- Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Odia, Punjabi
- Assamese, Maithili, Santali, Kashmiri, Sindhi, Dogri, Konkani, Bodo, Manipuri
- Sanskrit, Nepali, Urdu, English
"""
import os
import gc
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing language detection
try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed. Install: pip install langdetect")

# Try importing IndicTrans2
INDICTRANS_AVAILABLE = False
try:
    from IndicTransToolkit import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    INDICTRANS_AVAILABLE = True
except ImportError:
    logger.warning("IndicTransToolkit not installed. Translation will be disabled.")


@dataclass
class TranslationResult:
    """Result of translation operation."""
    original_text: str
    translated_text: str
    source_language: str
    source_lang_name: str
    was_translated: bool
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text[:200] + "..." if len(self.original_text) > 200 else self.original_text,
            "translated_text": self.translated_text[:200] + "..." if len(self.translated_text) > 200 else self.translated_text,
            "source_language": self.source_language,
            "source_lang_name": self.source_lang_name,
            "was_translated": self.was_translated,
            "success": self.success,
            "error_message": self.error_message
        }


class MultilingualProcessor:
    """
    Universal multilingual processor supporting ALL IndicTrans2 languages.
    
    Supports 22 Indian languages + English:
    - Detects language automatically
    - Translates to English if needed
    - Provides cultural context for visualization
    """
    
    # IndicTrans2 language codes mapping
    # ISO 639-1 -> IndicTrans2 code
    LANGUAGE_MAP = {
        # Major Indic Languages
        "hi": ("hin_Deva", "Hindi", "हिंदी"),
        "bn": ("ben_Beng", "Bengali", "বাংলা"),
        "ta": ("tam_Taml", "Tamil", "தமிழ்"),
        "te": ("tel_Telu", "Telugu", "తెలుగు"),
        "mr": ("mar_Deva", "Marathi", "मराठी"),
        "gu": ("guj_Gujr", "Gujarati", "ગુજરાતી"),
        "kn": ("kan_Knda", "Kannada", "ಕನ್ನಡ"),
        "ml": ("mal_Mlym", "Malayalam", "മലയാളം"),
        "or": ("ory_Orya", "Odia", "ଓଡ଼ିଆ"),
        "pa": ("pan_Guru", "Punjabi", "ਪੰਜਾਬੀ"),
        
        # Other Indic Languages
        "as": ("asm_Beng", "Assamese", "অসমীয়া"),
        "mai": ("mai_Deva", "Maithili", "मैथिली"),
        "sat": ("sat_Olck", "Santali", "ᱥᱟᱱᱛᱟᱲᱤ"),
        "ks": ("kas_Arab", "Kashmiri", "کٲشُر"),
        "sd": ("snd_Arab", "Sindhi", "سنڌي"),
        "doi": ("doi_Deva", "Dogri", "डोगरी"),
        "kok": ("kok_Deva", "Konkani", "कोंकणी"),
        "brx": ("brx_Deva", "Bodo", "बड़ो"),
        "mni": ("mni_Mtei", "Manipuri", "ꯃꯤꯇꯩꯂꯣꯟ"),
        "sa": ("san_Deva", "Sanskrit", "संस्कृतम्"),
        "ne": ("nep_Deva", "Nepali", "नेपाली"),
        "ur": ("urd_Arab", "Urdu", "اردو"),
    }
    
    # Cultural contexts for different regions
    CULTURAL_CONTEXTS = {
        "south_indian": (
            "South Indian setting, colorful traditional saree and veshti, "
            "temple gopurams and kolam patterns, warm golden lighting, "
            "tropical landscape with palm trees and paddy fields"
        ),
        "north_indian": (
            "North Indian setting, traditional salwar kameez and kurta, "
            "Mughal architecture with domes and arches, vibrant bazaar scenes, "
            "terracotta and jewel tones, bustling village atmosphere"
        ),
        "east_indian": (
            "Eastern Indian setting, traditional dhoti and saree, "
            "terracotta temples and bamboo groves, lotus ponds, "
            "monsoon greenery, artistic alpona patterns"
        ),
        "west_indian": (
            "Western Indian setting, vibrant Bandhani patterns, "
            "desert landscapes with camels, ornate havelis, "
            "colorful Rajasthani turbans, golden sandstone architecture"
        ),
        "northeast_indian": (
            "Northeast Indian setting, traditional tribal attire, "
            "lush green hills and forests, bamboo houses, "
            "vibrant festivals, traditional weaving patterns"
        ),
    }
    
    # Map languages to cultural regions
    LANGUAGE_REGIONS = {
        "ta": "south_indian", "te": "south_indian", "kn": "south_indian", "ml": "south_indian",
        "hi": "north_indian", "ur": "north_indian", "pa": "north_indian", "ks": "north_indian",
        "bn": "east_indian", "or": "east_indian", "as": "east_indian", "mai": "east_indian",
        "gu": "west_indian", "mr": "west_indian", "sd": "west_indian", "kok": "west_indian",
        "mni": "northeast_indian", "brx": "northeast_indian", "sat": "northeast_indian",
    }
    
    # Model configuration
    MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "indictrans")
    MIN_RAM_GB = 4
    
    def __init__(self, auto_load: bool = False):
        """Initialize processor. Set auto_load=True to load model immediately."""
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._device = None
        self._is_loaded = False
        
        if auto_load and INDICTRANS_AVAILABLE:
            self._load_model()
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names."""
        return {code: info[1] for code, info in self.LANGUAGE_MAP.items()}
    
    def _check_memory(self) -> Tuple[bool, float]:
        """Check if sufficient memory is available."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)
            return available_gb >= self.MIN_RAM_GB, available_gb
        except:
            return True, 8.0
    
    def _load_model(self) -> bool:
        """Load IndicTrans2 model for Indic → English translation."""
        if self._is_loaded:
            return True
        
        if not INDICTRANS_AVAILABLE:
            logger.error("IndicTransToolkit not available")
            return False
        
        has_memory, available_gb = self._check_memory()
        if not has_memory:
            logger.warning(f"Low memory: {available_gb:.1f}GB available")
            return False
        
        try:
            logger.info(f"Loading IndicTrans2 model...")
            os.makedirs(self.MODEL_DIR, exist_ok=True)
            
            import torch
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self._device = torch.device("cpu")
                logger.info("Using CPU for translation")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME,
                trust_remote_code=True,
                cache_dir=self.MODEL_DIR
            )
            
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.MODEL_NAME,
                trust_remote_code=True,
                cache_dir=self.MODEL_DIR,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self._device)
            
            self._processor = IndicProcessor(inference=True)
            self._is_loaded = True
            logger.info("IndicTrans2 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def detect_language(self, text: str) -> Tuple[str, str]:
        """
        Detect language of text.
        
        Returns: (iso_code, language_name)
        """
        if not LANGDETECT_AVAILABLE:
            return "en", "English"
        
        if not text or len(text.strip()) < 5:
            return "en", "English"
        
        try:
            lang = detect(text.strip()[:500])
            if lang in self.LANGUAGE_MAP:
                return lang, self.LANGUAGE_MAP[lang][1]
            elif lang == "en":
                return "en", "English"
            else:
                # Try to match similar language codes
                for code in self.LANGUAGE_MAP:
                    if lang.startswith(code) or code.startswith(lang):
                        return code, self.LANGUAGE_MAP[code][1]
                return lang, f"Unknown ({lang})"
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en", "English"
    
    def is_indic(self, text: str) -> bool:
        """Check if text is in an Indic language (not English)."""
        lang, _ = self.detect_language(text)
        return lang in self.LANGUAGE_MAP
    
    def translate(self, text: str, source_lang_code: str) -> TranslationResult:
        """
        Translate Indic language text to English.
        
        Args:
            text: Source text
            source_lang_code: ISO 639-1 code (e.g., 'ta', 'hi', 'bn')
        """
        if source_lang_code not in self.LANGUAGE_MAP:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_lang_code,
                source_lang_name="Unknown",
                was_translated=False,
                success=True,
                error_message=None
            )
        
        indic_code, lang_name, _ = self.LANGUAGE_MAP[source_lang_code]
        
        if not INDICTRANS_AVAILABLE:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_lang_code,
                source_lang_name=lang_name,
                was_translated=False,
                success=False,
                error_message="IndicTransToolkit not installed"
            )
        
        if not self._is_loaded and not self._load_model():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_lang_code,
                source_lang_name=lang_name,
                was_translated=False,
                success=False,
                error_message="Failed to load translation model"
            )
        
        try:
            import torch
            logger.info(f"Translating {lang_name} text ({len(text)} chars)...")
            
            batch = self._processor.preprocess_batch(
                [text],
                src_lang=indic_code,
                tgt_lang="eng_Latn"
            )
            
            inputs = self._tokenizer(
                batch,
                truncation=True,
                padding="longest",
                max_length=512,
                return_tensors="pt"
            ).to(self._device)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    num_beams=5,
                    num_return_sequences=1,
                    max_length=512
                )
            
            translated = self._tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            translated = self._processor.postprocess_batch(translated, lang="eng_Latn")
            translated_text = translated[0] if translated else text
            
            logger.info(f"Translation successful: {translated_text[:50]}...")
            
            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_lang_code,
                source_lang_name=lang_name,
                was_translated=True,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_lang_code,
                source_lang_name=lang_name,
                was_translated=False,
                success=False,
                error_message=str(e)
            )
    
    def process_text(self, text: str) -> TranslationResult:
        """
        Auto-detect language and translate if Indic.
        
        Returns: TranslationResult with English text
        """
        lang_code, lang_name = self.detect_language(text)
        
        if lang_code == "en" or lang_code not in self.LANGUAGE_MAP:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=lang_code,
                source_lang_name=lang_name,
                was_translated=False,
                success=True
            )
        
        logger.info(f"Detected {lang_name}, translating to English...")
        return self.translate(text, lang_code)
    
    def get_cultural_context(self, lang_code: str) -> str:
        """Get cultural context for the given language."""
        region = self.LANGUAGE_REGIONS.get(lang_code, "north_indian")
        return self.CULTURAL_CONTEXTS.get(region, self.CULTURAL_CONTEXTS["north_indian"])
    
    def enhance_prompt_with_culture(self, prompt: str, lang_code: str) -> str:
        """Add cultural context to an image generation prompt."""
        if lang_code in self.LANGUAGE_MAP:
            context = self.get_cultural_context(lang_code)
            return f"{prompt}, {context}"
        return prompt


# Backwards compatibility with TamilProcessor
class TamilProcessor(MultilingualProcessor):
    """Backwards-compatible alias for MultilingualProcessor."""
    
    SOUTH_INDIAN_CONTEXT = MultilingualProcessor.CULTURAL_CONTEXTS["south_indian"]
    
    def is_tamil(self, text: str) -> bool:
        lang, _ = self.detect_language(text)
        return lang == "ta"
    
    def get_cultural_context(self) -> str:
        return self.SOUTH_INDIAN_CONTEXT


# Singleton instances
_processor: Optional[MultilingualProcessor] = None
_tamil_processor: Optional[TamilProcessor] = None


def get_multilingual_processor() -> MultilingualProcessor:
    """Get singleton MultilingualProcessor."""
    global _processor
    if _processor is None:
        _processor = MultilingualProcessor(auto_load=False)
    return _processor


def get_tamil_processor() -> TamilProcessor:
    """Get singleton TamilProcessor (backwards compatible)."""
    global _tamil_processor
    if _tamil_processor is None:
        _tamil_processor = TamilProcessor(auto_load=False)
    return _tamil_processor


# Test
if __name__ == "__main__":
    processor = MultilingualProcessor()
    
    # Test texts in different languages
    tests = [
        ("ஒரு சிறிய கிராமத்தில் மாயா என்ற பெண் வாழ்ந்தாள்.", "Tamil"),
        ("हिंदी में एक छोटी कहानी लिखें।", "Hindi"),
        ("একটি ছোট গল্প লেখ।", "Bengali"),
        ("Machine learning is a subset of AI.", "English"),
    ]
    
    print("=" * 60)
    print("MULTILINGUAL PROCESSOR TEST")
    print("=" * 60)
    print(f"Supported languages: {len(processor.get_supported_languages())}")
    print()
    
    for text, expected in tests:
        lang, name = processor.detect_language(text)
        print(f"Text: {text[:30]}...")
        print(f"  Detected: {name} ({lang})")
        print()
