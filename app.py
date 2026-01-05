"""
VisualVerse - Streamlit Application
A dual-mode NLP system for converting text to Comics or Mind-Maps.
"""
import streamlit as st
import base64
from pathlib import Path
import io
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.classifier import TextClassifier, TextType
from core.router import PipelineRouter

from pipelines.comic.segmenter import StorySegmenter
from pipelines.comic.extractor import SceneExtractor
from pipelines.comic.prompt_builder import PromptBuilder
from pipelines.comic.image_generator import ImageGenerator
from pipelines.comic.layout import ComicLayout

from pipelines.mindmap.keyphrase import KeyphraseExtractor
from pipelines.mindmap.relation_extractor import RelationExtractor
from pipelines.mindmap.graph_builder import GraphBuilder
from pipelines.mindmap.visualizer import MindMapVisualizer


# Page configuration
st.set_page_config(
    page_title="VisualVerse",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium futuristic styling
st.markdown("""
<style>
    /* ===== IMPORTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ===== ROOT VARIABLES ===== */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-tertiary: #1a1a2e;
        --accent-cyan: #00f5ff;
        --accent-purple: #b829ff;
        --accent-pink: #ff2d7a;
        --accent-blue: #4361ee;
        --text-primary: #ffffff;
        --text-secondary: rgba(255,255,255,0.7);
        --text-muted: rgba(255,255,255,0.4);
        --glass-bg: rgba(255,255,255,0.03);
        --glass-border: rgba(255,255,255,0.08);
        --glow-cyan: 0 0 20px rgba(0,245,255,0.5);
        --glow-purple: 0 0 20px rgba(184,41,255,0.5);
        --glow-pink: 0 0 20px rgba(255,45,122,0.5);
    }
    
    /* ===== GLOBAL STYLES ===== */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-tertiary) 50%, #0f0f23 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-cyan), var(--accent-purple));
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--accent-purple), var(--accent-pink));
    }
    
    /* ===== ANIMATED HEADER ===== */
    .main-header {
        font-family: 'Orbitron', monospace !important;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink), var(--accent-cyan));
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        animation: gradient-shift 4s ease infinite;
        text-shadow: 0 0 40px rgba(0,245,255,0.3);
        letter-spacing: 4px;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif !important;
        text-align: center;
        color: var(--text-secondary) !important;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    /* ===== GLASSMORPHISM SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(18,18,26,0.95) 0%, rgba(10,10,15,0.98) 100%) !important;
        border-right: 1px solid var(--glass-border) !important;
        backdrop-filter: blur(20px) !important;
    }
    
    section[data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 2px;
        height: 100%;
        background: linear-gradient(180deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink));
        animation: border-glow 3s ease-in-out infinite;
    }
    
    @keyframes border-glow {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Orbitron', monospace !important;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* ===== RADIO BUTTONS (MODE SELECTOR) ===== */
    div[data-testid="stRadio"] > label {
        font-family: 'Orbitron', monospace !important;
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 8px;
    }
    
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        backdrop-filter: blur(10px);
    }
    
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--glow-cyan);
        transform: translateX(5px);
    }
    
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(0,245,255,0.15), rgba(184,41,255,0.15)) !important;
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--glow-cyan);
    }
    
    /* ===== SELECT BOXES ===== */
    div[data-testid="stSelectbox"] > div > div {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stSelectbox"] > div > div:hover {
        border-color: var(--accent-purple) !important;
        box-shadow: var(--glow-purple);
    }
    
    /* ===== SLIDERS ===== */
    div[data-testid="stSlider"] > div > div > div {
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple)) !important;
    }
    
    div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
        background: var(--accent-cyan) !important;
        color: var(--bg-primary) !important;
        font-weight: 600;
    }
    
    /* ===== TEXT AREA ===== */
    .stTextArea textarea {
        background: rgba(10,10,15,0.8) !important;
        border: 2px solid var(--glass-border) !important;
        border-radius: 16px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
        padding: 16px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 30px rgba(0,245,255,0.3), inset 0 0 20px rgba(0,245,255,0.05) !important;
        outline: none !important;
    }
    
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        font-family: 'Orbitron', monospace !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 1px;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
        color: var(--bg-primary) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(0,245,255,0.3);
        text-transform: uppercase;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 30px rgba(0,245,255,0.5), 0 0 60px rgba(184,41,255,0.3) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98) !important;
    }
    
    /* Primary button special styling */
    button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-pink), var(--accent-purple)) !important;
        box-shadow: 0 4px 20px rgba(255,45,122,0.4) !important;
    }
    
    button[kind="primary"]:hover {
        box-shadow: 0 8px 40px rgba(255,45,122,0.6), 0 0 80px rgba(184,41,255,0.4) !important;
    }
    
    /* ===== DOWNLOAD BUTTONS ===== */
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(0,245,255,0.1), rgba(184,41,255,0.1)) !important;
        border: 2px solid var(--accent-cyan) !important;
        color: var(--accent-cyan) !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
        color: var(--bg-primary) !important;
    }
    
    /* ===== CLASSIFICATION BOXES ===== */
    .classification-box {
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        color: white;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
    }
    
    .classification-box::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink), var(--accent-cyan));
        background-size: 400% 400%;
        z-index: -1;
        border-radius: 22px;
        animation: gradient-border 4s ease infinite;
    }
    
    .classification-box::after {
        content: '';
        position: absolute;
        top: 2px;
        left: 2px;
        right: 2px;
        bottom: 2px;
        background: var(--bg-secondary);
        border-radius: 18px;
        z-index: -1;
    }
    
    @keyframes gradient-border {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .classification-box h4 {
        font-family: 'Orbitron', monospace !important;
        color: var(--accent-cyan) !important;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .classification-box p {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-secondary) !important;
        margin: 0.4rem 0;
    }
    
    .classification-box p strong {
        color: var(--text-primary) !important;
    }
    
    .narrative-box {
        background: linear-gradient(135deg, rgba(255,45,122,0.15), rgba(184,41,255,0.1));
    }
    
    .narrative-box h4 {
        color: var(--accent-pink) !important;
    }
    
    .informational-box {
        background: linear-gradient(135deg, rgba(0,245,255,0.15), rgba(67,97,238,0.1));
    }
    
    /* ===== SUBHEADERS ===== */
    .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Orbitron', monospace !important;
        color: var(--text-primary) !important;
        position: relative;
        display: inline-block;
    }
    
    .main h2::after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 50px;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
        border-radius: 2px;
    }
    
    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        font-family: 'Orbitron', monospace !important;
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: var(--accent-purple) !important;
        box-shadow: var(--glow-purple);
    }
    
    .streamlit-expanderContent {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    /* ===== ALERTS / INFO BOXES ===== */
    .stAlert {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
    }
    
    div[data-testid="stAlert"] {
        border-left: 4px solid var(--accent-cyan) !important;
    }
    
    /* ===== SPINNER ===== */
    .stSpinner > div {
        border-color: var(--accent-cyan) transparent transparent transparent !important;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink)) !important;
        background-size: 200% 100%;
        animation: progress-glow 2s ease infinite;
    }
    
    @keyframes progress-glow {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    
    /* ===== DIVIDERS ===== */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, var(--accent-purple), transparent) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* ===== CHECKBOXES ===== */
    .stCheckbox label {
        color: var(--text-secondary) !important;
    }
    
    .stCheckbox label:hover {
        color: var(--text-primary) !important;
    }
    
    /* ===== CODE BLOCKS ===== */
    .stCodeBlock {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
    }
    
    /* ===== IMAGE CONTAINER ===== */
    .stImage {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    }
    
    /* ===== COLUMNS GAP ===== */
    div[data-testid="column"] {
        padding: 0 1rem;
    }
    
    /* ===== ANIMATED BACKGROUND ELEMENTS ===== */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 80%, rgba(0,245,255,0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(184,41,255,0.05) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(255,45,122,0.03) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* ===== TOOLTIP STYLING ===== */
    .stTooltipIcon {
        color: var(--accent-cyan) !important;
    }
    
    /* ===== SUCCESS/WARNING/ERROR ===== */
    .stSuccess {
        background: linear-gradient(135deg, rgba(0,245,255,0.1), rgba(0,255,136,0.1)) !important;
        border: 1px solid rgba(0,255,136,0.3) !important;
        border-radius: 12px !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255,193,7,0.1), rgba(255,152,0,0.1)) !important;
        border: 1px solid rgba(255,193,7,0.3) !important;
        border-radius: 12px !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(255,45,122,0.1), rgba(255,82,82,0.1)) !important;
        border: 1px solid rgba(255,45,122,0.3) !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = TextClassifier(use_semantic=False)
if 'router' not in st.session_state:
    st.session_state.router = PipelineRouter(st.session_state.classifier)
if 'last_result' not in st.session_state:
    st.session_state.last_result = None


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">⚡ VISUALVERSE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Next-Gen AI Text Visualization • Comics & Mind-Maps</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        mode = st.radio(
            "Visualization Mode",
            ["Comic", "Mind-Map"],
            help="Choose how to visualize your text"
        )
        
        st.divider()
        
        st.subheader("Comic Settings")
        
        # Model selection for local GPU
        model_choice = st.selectbox(
            "AI Model",
            ["dreamshaper", "cartoon", "sd15"],
            help="DreamShaper: Best for comic art | Cartoon: Anime style | SD1.5: Classic"
        )
        st.session_state.model_choice = model_choice
        
        comic_style = st.selectbox(
            "Art Style",
            ["western", "cartoon", "manga", "realistic"],
            help="Style of comic art"
        )
        max_panels = st.slider("Max Panels", 2, 6, 4)
        
        use_placeholder = st.checkbox("Use Placeholder Images", value=False, 
                                      help="If checked, uses placeholder images instead of AI generation")
        
        if not use_placeholder:
            st.info("🖥️ Using LOCAL GPU for image generation")
            # Check GPU
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    st.success(f"✅ GPU: {gpu_name}")
                else:
                    st.warning("⚠️ No GPU detected - will use CPU (slower)")
            except Exception:
                st.warning("⚠️ PyTorch not loaded yet")
        
        st.divider()
        
        st.subheader("Mind-Map Settings")
        max_keywords = st.slider("Max Keywords", 5, 25, 15)
        theme = st.selectbox("Theme", ["light", "dark"])
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Text")
        
        # Sample text buttons
        sample_col1, sample_col2 = st.columns(2)
        with sample_col1:
            if st.button("📖 Load Story Sample"):
                st.session_state.sample_text = """Once upon a time, in a small village nestled between two mountains, there lived a young girl named Maya. She had always dreamed of exploring the world beyond the peaks.

One sunny morning, Maya discovered a hidden path behind the old oak tree. "This must lead somewhere magical!" she exclaimed with excitement.

She followed the path deeper into the woods. The trees grew taller and the sunlight filtered through the leaves creating dancing shadows.

Suddenly, she arrived at a clearing. In the center stood an ancient stone tower covered in ivy. A mysterious light glowed from the highest window. Maya knew her adventure was just beginning."""
        
        with sample_col2:
            if st.button("📚 Load Info Sample"):
                st.session_state.sample_text = """Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn from and make predictions based on data.

It includes three main types: supervised learning uses labeled data to train models, unsupervised learning discovers hidden patterns in unlabeled data, and reinforcement learning uses rewards to guide decision-making.

Deep learning, a specialized form of machine learning, uses neural networks with multiple layers to model complex patterns. Applications include natural language processing, computer vision, and speech recognition.

Therefore, machine learning has become essential for modern AI systems, enabling everything from recommendation engines to autonomous vehicles."""
        
        # Text input
        default_text = st.session_state.get('sample_text', '')
        user_text = st.text_area(
            "Enter your text here:",
            value=default_text,
            height=300,
            placeholder="Paste or type your text here... (minimum 50 characters)"
        )
        
        # Process button
        if st.button("🚀 Generate Visualization", type="primary"):
            if len(user_text) < 50:
                st.error("Please enter at least 50 characters of text.")
            else:
                process_text(user_text, mode, comic_style, max_panels, 
                           use_placeholder, max_keywords, theme)
    
    with col2:
        st.subheader("🎨 Output")
        display_results()


def process_text(text, mode, comic_style, max_panels, use_placeholder, max_keywords, theme):
    """Process text and generate visualization."""
    with st.spinner("Analyzing text..."):
        # Classify text (still classify for display purposes)
        result = st.session_state.classifier.classify(text)
        
        # Determine pipeline based on user selection
        if mode == "Comic":
            pipeline = "comic"
        else:  # Mind-Map
            pipeline = "mindmap"
        
        st.session_state.classification = result
        st.session_state.pipeline = pipeline
    
    # Generate visualization
    if pipeline == "comic":
        generate_comic(text, comic_style, max_panels, use_placeholder)
    else:
        generate_mindmap(text, max_keywords, theme)


def generate_comic(text, style, max_panels, use_placeholder):
    """Generate comic from narrative text."""
    with st.spinner("Segmenting story into scenes..."):
        segmenter = StorySegmenter(max_segments=max_panels)
        segments = segmenter.segment(text)
        st.session_state.segments = segments
    
    with st.spinner("Extracting scene details..."):
        extractor = SceneExtractor()
        scenes = [extractor.extract(seg.id, seg.text) for seg in segments]
        st.session_state.scenes = scenes
    
    with st.spinner("Building image prompts..."):
        prompt_builder = PromptBuilder()
        prompt_builder.set_comic_style(style)
        prompts = prompt_builder.build_prompts(scenes)
        st.session_state.prompts = prompts
    
    # Get model choice from session state
    model_choice = st.session_state.get('model_choice', 'dreamshaper')
    
    if use_placeholder:
        with st.spinner("Generating placeholder panels..."):
            generator = ImageGenerator(use_placeholder=True)
            images = generator.generate_batch(prompts)
            st.session_state.images = images
    else:
        st.info(f"🖥️ Generating {len(prompts)} comic panels on LOCAL GPU...")
        st.warning("⏳ First run downloads the model (~4GB). Subsequent runs are faster.")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Use local GPU model
        generator = ImageGenerator(
            model_name=model_choice,
            use_placeholder=False
        )
        
        images = []
        for i, prompt in enumerate(prompts):
            status_text.text(f"Generating panel {i+1}/{len(prompts)}...")
            progress_bar.progress((i) / len(prompts))
            
            image = generator.generate(prompt)
            images.append(image)
            
            if image.success:
                st.success(f"✓ Panel {i+1} generated!")
            else:
                st.warning(f"Panel {i+1} failed: {image.error}")
        
        progress_bar.progress(1.0)
        status_text.text("All panels generated!")
        st.session_state.images = images
    
    with st.spinner("Assembling comic strip..."):
        layout = ComicLayout()
        # Filter to only successful images
        valid_images = [img for img in st.session_state.images if img.success]
        if valid_images:
            columns = min(3, len(valid_images))
            comic = layout.create_strip(valid_images, columns=columns, title="Generated Comic")
            st.session_state.comic = comic
            st.session_state.last_result = "comic"
        else:
            st.error("No images were generated successfully.")
            st.session_state.last_result = None


def generate_mindmap(text, max_keywords, theme):
    """Generate mind map from informational text."""
    with st.spinner("Extracting keyphrases..."):
        keyphrase_extractor = KeyphraseExtractor(max_keywords=max_keywords)
        keyphrases = keyphrase_extractor.extract(text)
        st.session_state.keyphrases = keyphrases
    
    with st.spinner("Analyzing relationships..."):
        relation_extractor = RelationExtractor()
        keyphrase_strings = [kp.phrase for kp in keyphrases]
        relations = relation_extractor.extract(text, keyphrase_strings)
        st.session_state.relations = relations
    
    with st.spinner("Building concept graph..."):
        graph_builder = GraphBuilder()
        graph = graph_builder.build_from_keyphrases_and_relations(keyphrases, relations)
        st.session_state.graph_builder = graph_builder
    
    with st.spinner("Rendering mind map..."):
        visualizer = MindMapVisualizer()
        visualizer.set_theme(theme)
        network = visualizer.visualize(graph_builder, "Mind Map")
        html_content = visualizer.get_html_string()
        st.session_state.mindmap_html = html_content
        st.session_state.last_result = "mindmap"


def display_results():
    """Display generation results."""
    # Display classification
    if 'classification' in st.session_state:
        result = st.session_state.classification
        
        box_class = "narrative-box" if result.text_type == TextType.NARRATIVE else "informational-box"
        icon = "📖" if result.text_type == TextType.NARRATIVE else "📊"
        
        st.markdown(f"""
        <div class="classification-box {box_class}">
            <h4>{icon} Text Classification</h4>
            <p><strong>Type:</strong> {result.text_type.value.upper()}</p>
            <p><strong>Confidence:</strong> {result.confidence:.1%}</p>
            <p><strong>Narrative Score:</strong> {result.narrative_score:.2f} | 
               <strong>Informational Score:</strong> {result.informational_score:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display Comic Result
    if st.session_state.last_result == "comic" and 'comic' in st.session_state:
        st.subheader("📚 Generated Comic Strip")
        
        # Display comic image
        comic = st.session_state.comic
        img_bytes = io.BytesIO(comic.image_data)
        st.image(img_bytes, caption=f"Comic Strip ({comic.panel_count} panels)", 
                use_container_width=True)
        
        # Download button
        st.download_button(
            label="⬇️ Download Comic",
            data=comic.image_data,
            file_name="visualverse_comic.png",
            mime="image/png"
        )
        
        # Scene details expander
        with st.expander("📋 Scene Details"):
            for i, scene in enumerate(st.session_state.scenes):
                st.markdown(f"**Panel {i+1}**")
                st.write(f"- Characters: {[c.name for c in scene.characters]}")
                st.write(f"- Setting: {scene.setting.location}")
                st.write(f"- Mood: {scene.mood}")
                if scene.dialogue:
                    st.write(f"- Dialogue: {scene.dialogue}")
                st.divider()
        
        # Prompts expander
        with st.expander("🎨 Image Prompts"):
            for i, prompt in enumerate(st.session_state.prompts):
                st.markdown(f"**Panel {i+1}**")
                st.code(prompt.get_full_prompt())
    
    # Display Mind Map Result
    elif st.session_state.last_result == "mindmap" and 'mindmap_html' in st.session_state:
        st.subheader("🗺️ Generated Mind Map")
        
        # Display interactive mind map
        st.components.v1.html(st.session_state.mindmap_html, height=600, scrolling=True)
        
        # Download buttons in columns
        dl_col1, dl_col2 = st.columns(2)
        
        with dl_col1:
            st.download_button(
                label="⬇️ Download HTML",
                data=st.session_state.mindmap_html,
                file_name="visualverse_mindmap.html",
                mime="text/html"
            )
        
        with dl_col2:
            # Generate PNG
            if 'graph_builder' in st.session_state:
                visualizer = MindMapVisualizer()
                png_data = visualizer.generate_png(st.session_state.graph_builder)
                if png_data:
                    st.download_button(
                        label="🖼️ Download PNG",
                        data=png_data,
                        file_name="visualverse_mindmap.png",
                        mime="image/png"
                    )
        
        # Keywords expander
        with st.expander("🔑 Extracted Keyphrases"):
            for kp in st.session_state.keyphrases:
                st.write(f"- **{kp.phrase}** (score: {kp.score:.3f}, source: {kp.source})")
        
        # Relations expander
        with st.expander("🔗 Discovered Relationships"):
            for rel in st.session_state.relations:
                st.write(f"- {rel.source} → *{rel.relation_type.value}* → {rel.target}")
        
        # Graph stats
        if 'graph_builder' in st.session_state:
            gb = st.session_state.graph_builder
            st.info(f"📊 Graph Statistics: {len(gb.nodes)} nodes, {len(gb.edges)} edges")
    
    else:
        st.info("👆 Enter text and click 'Generate Visualization' to see results")


if __name__ == "__main__":
    main()
