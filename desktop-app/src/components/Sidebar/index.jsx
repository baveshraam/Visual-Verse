import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './Sidebar.css';

export default function Sidebar({
    mode,
    settings,
    onSettingsChange,
    backendStatus,
    gpuInfo
}) {
    const [expandedSection, setExpandedSection] = useState('comic');

    const toggleSection = (section) => {
        setExpandedSection(expandedSection === section ? null : section);
    };

    // GPU status display
    const gpuStatus = gpuInfo?.available
        ? `${gpuInfo.device}`
        : 'CPU Mode';
    const gpuStatusClass = gpuInfo?.available ? 'online' : 'offline';

    return (
        <aside className="sidebar">
            {/* Header */}
            <div className="sidebar-header">
                <h1 className="app-title gradient-text">⚡ VISUALVERSE</h1>
                <p className="app-subtitle">Next-Gen AI Text Visualization</p>
            </div>

            <div className="divider" />

            {/* Backend Status */}
            <div className="status-section">
                <div className="status-row">
                    <span className="status-label">Backend</span>
                    <div className="status-badge">
                        <span className={`status-dot ${backendStatus}`}></span>
                        <span className="status-text">
                            {backendStatus === 'online' ? 'Connected' :
                                backendStatus === 'processing' ? 'Processing...' : 'Offline'}
                        </span>
                    </div>
                </div>
                <div className="status-row">
                    <span className="status-label">GPU</span>
                    <div className="status-badge">
                        <span className={`status-dot ${gpuStatusClass}`}></span>
                        <span className="status-text" title={gpuInfo?.memory || ''}>
                            {gpuInfo?.available ? '✓ Available' : 'Not Available'}
                        </span>
                    </div>
                </div>
            </div>

            <div className="divider" />

            {/* Settings Sections */}
            <div className="settings-container">
                {/* Comic Settings */}
                <SettingsSection
                    title="Comic Settings"
                    icon="🎨"
                    isExpanded={expandedSection === 'comic'}
                    onToggle={() => toggleSection('comic')}
                >
                    <div className="setting-item">
                        <label className="setting-label">Art Style</label>
                        <select
                            className="setting-select"
                            value={settings.comicStyle}
                            onChange={(e) => onSettingsChange('comicStyle', e.target.value)}
                        >
                            <option value="western">Western</option>
                            <option value="cartoon">Cartoon</option>
                            <option value="manga">Manga</option>
                            <option value="realistic">Realistic</option>
                        </select>
                    </div>

                    <div className="setting-item">
                        <label className="setting-label">AI Model</label>
                        <select
                            className="setting-select"
                            value={settings.model}
                            onChange={(e) => onSettingsChange('model', e.target.value)}
                        >
                            <option value="dreamshaper">DreamShaper (Best)</option>
                            <option value="cartoon">Cartoon Style</option>
                            <option value="sd15">Stable Diffusion 1.5</option>
                        </select>
                    </div>

                    <div className="setting-item">
                        <label className="setting-label">
                            Max Panels: <span className="setting-value">{settings.maxPanels}</span>
                        </label>
                        <input
                            type="range"
                            className="setting-slider"
                            min="2"
                            max="6"
                            value={settings.maxPanels}
                            onChange={(e) => onSettingsChange('maxPanels', parseInt(e.target.value))}
                        />
                    </div>

                    <div className="setting-item">
                        <label className="setting-checkbox">
                            <input
                                type="checkbox"
                                checked={settings.usePlaceholder}
                                onChange={(e) => onSettingsChange('usePlaceholder', e.target.checked)}
                            />
                            <span className="checkbox-custom"></span>
                            Use Placeholder Images
                        </label>
                        <span className="setting-hint">Skip AI generation for faster preview</span>
                    </div>
                </SettingsSection>

                {/* Mind-Map Settings */}
                <SettingsSection
                    title="Mind-Map Settings"
                    icon="🧠"
                    isExpanded={expandedSection === 'mindmap'}
                    onToggle={() => toggleSection('mindmap')}
                >
                    <div className="setting-item">
                        <label className="setting-label">
                            Max Keywords: <span className="setting-value">{settings.maxKeywords}</span>
                        </label>
                        <input
                            type="range"
                            className="setting-slider"
                            min="5"
                            max="25"
                            value={settings.maxKeywords}
                            onChange={(e) => onSettingsChange('maxKeywords', parseInt(e.target.value))}
                        />
                    </div>

                    <div className="setting-item">
                        <label className="setting-label">Theme</label>
                        <select
                            className="setting-select"
                            value={settings.theme}
                            onChange={(e) => onSettingsChange('theme', e.target.value)}
                        >
                            <option value="dark">Dark</option>
                            <option value="light">Light</option>
                        </select>
                    </div>
                </SettingsSection>
            </div>

            {/* Footer */}
            <div className="sidebar-footer">
                <div className={`gpu-status ${gpuStatusClass}`}>
                    <span className="gpu-icon">{gpuInfo?.available ? '🟢' : '🔴'}</span>
                    <span className="gpu-text">{gpuStatus}</span>
                    {gpuInfo?.memory && <span className="gpu-memory">({gpuInfo.memory})</span>}
                </div>
            </div>
        </aside>
    );
}

// Collapsible Settings Section
function SettingsSection({ title, icon, isExpanded, onToggle, children }) {
    return (
        <div className="settings-section">
            <button className="section-header" onClick={onToggle}>
                <span className="section-icon">{icon}</span>
                <span className="section-title">{title}</span>
                <motion.span
                    className="section-arrow"
                    animate={{ rotate: isExpanded ? 180 : 0 }}
                >
                    ▼
                </motion.span>
            </button>

            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        className="section-content"
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                    >
                        {children}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
