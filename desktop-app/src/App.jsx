import { useState, useCallback } from 'react';
import { AnimatePresence } from 'framer-motion';

// Components
import Background3D from './components/Background3D';
import Sidebar from './components/Sidebar';
import TextInput from './components/TextInput';
import ModeSelector from './components/ModeSelector';
import ComicDisplay from './components/ComicDisplay';
import MindMapView from './components/MindMapView';
import LoadingOverlay from './components/LoadingOverlay';

// Hooks
import { useApi } from './hooks/useApi';

// Styles
import './App.css';

function App() {
    // API connection
    const api = useApi();

    // App state
    const [text, setText] = useState('');
    const [mode, setMode] = useState('comic');
    const [isGenerating, setIsGenerating] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState('');

    // Results
    const [comicResult, setComicResult] = useState(null);
    const [mindmapResult, setMindmapResult] = useState(null);
    const [classification, setClassification] = useState(null);

    // Progress
    const [progress, setProgress] = useState({ status: '', percent: 0 });

    // Settings
    const [settings, setSettings] = useState({
        comicStyle: 'western',
        model: 'dreamshaper',
        maxPanels: 4,
        usePlaceholder: false,  // Set to false for real AI image generation
        maxKeywords: 15,
        theme: 'dark'
    });

    const handleSettingsChange = useCallback((key, value) => {
        setSettings(prev => ({ ...prev, [key]: value }));
    }, []);

    const handleGenerate = useCallback(async () => {
        if (text.length < 50) return;

        setIsGenerating(true);
        setLoadingMessage('Analyzing text...');
        setProgress({ status: 'Analyzing text...', percent: 10 });

        try {
            // First classify the text
            const classResult = await api.classifyText(text);
            setClassification(classResult);
            setProgress({ status: 'Text classified', percent: 20 });

            if (mode === 'comic') {
                // Generate comic
                setLoadingMessage('Segmenting story into scenes...');
                setProgress({ status: 'Segmenting story...', percent: 30 });

                await new Promise(r => setTimeout(r, 500)); // Small delay for UI update
                setLoadingMessage('Extracting scene details...');
                setProgress({ status: 'Extracting scene details...', percent: 40 });

                await new Promise(r => setTimeout(r, 500));
                setLoadingMessage('Loading AI model (first run downloads ~2GB)...');
                setProgress({ status: 'Loading Stable Diffusion model...', percent: 50 });

                const result = await api.generateComic(text, {
                    style: settings.comicStyle,
                    maxPanels: settings.maxPanels,
                    usePlaceholder: settings.usePlaceholder
                });

                console.log('[DEBUG] Comic API Response:', Object.keys(result));

                setLoadingMessage('Rendering final comic panels...');
                setProgress({ status: 'Rendering comic panels...', percent: 80 });

                // Map API response to frontend format (snake_case → camelCase)
                setComicResult({
                    imageData: result.comic_image_base64,  // API: comic_image_base64
                    panelCount: result.panel_count,         // API: panel_count
                    scenes: result.scenes || [],            // API: scenes
                    segments: result.segments || [],        // API: segments
                    prompts: result.prompts || []           // API: prompts
                });
                setProgress({ status: 'Comic generated!', percent: 100 });

            } else {
                // Generate mind map
                setLoadingMessage('Extracting keyphrases...');
                setProgress({ status: 'Extracting concepts...', percent: 30 });

                await new Promise(r => setTimeout(r, 500));
                setProgress({ status: 'Finding relationships...', percent: 50 });

                const result = await api.generateMindMap(text, {
                    maxKeywords: settings.maxKeywords,
                    theme: settings.theme
                });

                console.log('[DEBUG] Mindmap API Response:', Object.keys(result));

                setProgress({ status: 'Building visualization...', percent: 80 });

                // Map API response to frontend format
                setMindmapResult({
                    keyphrases: result.keyphrases || [],
                    relations: result.relations || [],
                    graph: result.graph || {},
                    hierarchy: result.hierarchy || {},
                    htmlContent: result.visualization_html  // API: visualization_html
                });
                setProgress({ status: 'Mind map generated!', percent: 100 });
            }

        } catch (error) {
            console.error('Generation failed:', error);
            setProgress({ status: `Error: ${error.message}`, percent: 0 });
        } finally {
            setIsGenerating(false);
            setLoadingMessage('');
        }
    }, [text, mode, settings, api]);

    // Determine backend status
    const backendStatus = api.isLoading ? 'processing' :
        api.isConnected ? 'online' : 'offline';

    return (
        <div className="app">
            {/* 3D Animated Background */}
            <Background3D />

            {/* Main Layout */}
            <div className="app-container">
                {/* Sidebar */}
                <Sidebar
                    mode={mode}
                    settings={settings}
                    onSettingsChange={handleSettingsChange}
                    backendStatus={backendStatus}
                    gpuInfo={api.gpuInfo}
                />

                {/* Main Content */}
                <main className="main-content">
                    <div className="content-wrapper">
                        {/* Left Panel - Input */}
                        <section className="input-section glass-card">
                            <ModeSelector
                                selected={mode}
                                onChange={setMode}
                            />

                            <div className="divider" />

                            <TextInput
                                value={text}
                                onChange={setText}
                                onGenerate={handleGenerate}
                                isLoading={isGenerating}
                            />

                            {/* Classification Result */}
                            {classification && (
                                <div className="classification-result glass-card">
                                    <h4>
                                        {classification.text_type === 'narrative' ? '📖' : '📊'}
                                        {' '}Text Analysis
                                    </h4>
                                    <div className="classification-details">
                                        <span className="detail">
                                            <strong>Type:</strong> {classification.text_type?.toUpperCase()}
                                        </span>
                                        <span className="detail">
                                            <strong>Confidence:</strong> {(classification.confidence * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                            )}
                        </section>

                        {/* Right Panel - Output */}
                        <section className="output-section glass-card">
                            {mode === 'comic' ? (
                                <ComicDisplay
                                    comic={comicResult}
                                    scenes={comicResult?.scenes}
                                    isGenerating={isGenerating && mode === 'comic'}
                                    progress={progress}
                                />
                            ) : (
                                <MindMapView
                                    mindmap={mindmapResult}
                                    keyphrases={mindmapResult?.keyphrases}
                                    relations={mindmapResult?.relations}
                                    isGenerating={isGenerating && mode === 'mindmap'}
                                />
                            )}
                        </section>
                    </div>
                </main>
            </div>

            {/* Loading Overlay */}
            <AnimatePresence>
                {isGenerating && (
                    <LoadingOverlay
                        isVisible={isGenerating}
                        message={loadingMessage}
                    />
                )}
            </AnimatePresence>
        </div>
    );
}

export default App;
