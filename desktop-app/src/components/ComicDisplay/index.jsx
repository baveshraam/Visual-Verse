import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './ComicDisplay.css';

export default function ComicDisplay({
    comic,
    scenes,
    isGenerating,
    progress
}) {
    const [selectedPanel, setSelectedPanel] = useState(null);
    const [showDetails, setShowDetails] = useState(false);

    if (!comic && !isGenerating) {
        return (
            <div className="comic-placeholder">
                <div className="placeholder-icon">📚</div>
                <h3>Comic Output</h3>
                <p>Enter a story and click Generate to create a comic strip</p>
            </div>
        );
    }

    return (
        <div className="comic-display">
            {/* Header */}
            <div className="comic-header">
                <h2 className="comic-title">
                    <span className="title-icon">🎨</span>
                    Generated Comic Strip
                </h2>
                {comic && (
                    <div className="comic-actions">
                        <motion.button
                            className="action-btn"
                            onClick={() => setShowDetails(!showDetails)}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                        >
                            {showDetails ? '✕ Hide Details' : '📋 Show Details'}
                        </motion.button>
                        <motion.button
                            className="action-btn primary"
                            onClick={() => downloadComic(comic)}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                        >
                            ⬇️ Download
                        </motion.button>
                    </div>
                )}
            </div>

            {/* Progress Bar during generation */}
            <AnimatePresence>
                {isGenerating && (
                    <motion.div
                        className="generation-progress"
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                    >
                        <div className="progress-info">
                            <span className="progress-text">{progress.status}</span>
                            <span className="progress-percent">{Math.round(progress.percent)}%</span>
                        </div>
                        <div className="progress-bar">
                            <motion.div
                                className="progress-fill"
                                initial={{ width: 0 }}
                                animate={{ width: `${progress.percent}%` }}
                                transition={{ duration: 0.3 }}
                            />
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Comic Image */}
            {comic && (
                <motion.div
                    className="comic-container"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, ease: 'easeOut' }}
                >
                    <div className="comic-frame glow-border">
                        <img
                            src={comic.imageUrl || `data:image/png;base64,${comic.imageData}`}
                            alt="Generated Comic Strip"
                            className="comic-image"
                        />
                    </div>

                    {comic.panelCount && (
                        <div className="comic-info">
                            <span className="panel-count">{comic.panelCount} Panels</span>
                        </div>
                    )}
                </motion.div>
            )}

            {/* Scene Details Panel */}
            <AnimatePresence>
                {showDetails && scenes && (
                    <motion.div
                        className="scene-details"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                    >
                        <h3 className="details-title">Scene Breakdown</h3>
                        <div className="scenes-grid">
                            {scenes.map((scene, index) => (
                                <motion.div
                                    key={index}
                                    className="scene-card glass-card"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: index * 0.1 }}
                                    whileHover={{ scale: 1.02, y: -5 }}
                                >
                                    <div className="scene-number">Panel {index + 1}</div>
                                    <div className="scene-content">
                                        {scene.characters && scene.characters.length > 0 && (
                                            <div className="scene-row">
                                                <span className="row-label">👤 Characters:</span>
                                                <span className="row-value">
                                                    {scene.characters.map(c => c.name || c).join(', ')}
                                                </span>
                                            </div>
                                        )}
                                        {scene.setting && (
                                            <div className="scene-row">
                                                <span className="row-label">📍 Setting:</span>
                                                <span className="row-value">
                                                    {scene.setting.location || scene.setting}
                                                </span>
                                            </div>
                                        )}
                                        {scene.mood && (
                                            <div className="scene-row">
                                                <span className="row-label">🎭 Mood:</span>
                                                <span className="row-value">{scene.mood}</span>
                                            </div>
                                        )}
                                        {scene.dialogue && (
                                            <div className="scene-row">
                                                <span className="row-label">💬 Dialogue:</span>
                                                <span className="row-value dialogue">"{scene.dialogue}"</span>
                                            </div>
                                        )}
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

function downloadComic(comic) {
    const link = document.createElement('a');
    if (comic.imageUrl) {
        link.href = comic.imageUrl;
    } else if (comic.imageData) {
        link.href = `data:image/png;base64,${comic.imageData}`;
    }
    link.download = 'visualverse_comic.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
