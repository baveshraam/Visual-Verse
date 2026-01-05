import { motion, AnimatePresence } from 'framer-motion';
import { useState, useEffect, useRef } from 'react';
import './LoadingOverlay.css';

// Simulate detailed log messages based on the current stage
const getLogMessages = (message) => {
    const logs = [];
    const timestamp = () => new Date().toLocaleTimeString();

    if (message.toLowerCase().includes('analyzing')) {
        logs.push({ time: timestamp(), text: '🔍 Initializing NLP pipeline...', type: 'info' });
        logs.push({ time: timestamp(), text: '📊 Loading text classifier model...', type: 'info' });
    } else if (message.toLowerCase().includes('segment')) {
        logs.push({ time: timestamp(), text: '✂️ Segmenting text into scenes...', type: 'info' });
        logs.push({ time: timestamp(), text: '🎭 Extracting narrative elements...', type: 'info' });
    } else if (message.toLowerCase().includes('generating') || message.toLowerCase().includes('render')) {
        logs.push({ time: timestamp(), text: '🚀 Loading Stable Diffusion model...', type: 'warning' });
        logs.push({ time: timestamp(), text: '⏳ First run may download ~2GB model from HuggingFace', type: 'warning' });
        logs.push({ time: timestamp(), text: '🎨 Model: Lykon/dreamshaper-8', type: 'info' });
    } else if (message.toLowerCase().includes('extract')) {
        logs.push({ time: timestamp(), text: '🔑 Extracting keyphrases...', type: 'info' });
        logs.push({ time: timestamp(), text: '🔗 Finding semantic relationships...', type: 'info' });
    }

    return logs;
};

export default function LoadingOverlay({ isVisible, message = 'Processing...', logs = [] }) {
    const [displayLogs, setDisplayLogs] = useState([]);
    const [showDetails, setShowDetails] = useState(true);
    const logContainerRef = useRef(null);
    const prevMessageRef = useRef('');

    // Add contextual logs when message changes
    useEffect(() => {
        if (message !== prevMessageRef.current) {
            const contextLogs = getLogMessages(message);
            if (contextLogs.length > 0) {
                setDisplayLogs(prev => [...prev, ...contextLogs].slice(-15)); // Keep last 15 logs
            }
            prevMessageRef.current = message;
        }
    }, [message]);

    // Add external logs
    useEffect(() => {
        if (logs.length > 0) {
            setDisplayLogs(prev => [...prev, ...logs].slice(-15));
        }
    }, [logs]);

    // Auto-scroll to bottom
    useEffect(() => {
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [displayLogs]);

    // Reset logs when overlay becomes visible
    useEffect(() => {
        if (isVisible) {
            setDisplayLogs([{
                time: new Date().toLocaleTimeString(),
                text: '🚀 Starting generation pipeline...',
                type: 'success'
            }]);
        }
    }, [isVisible]);

    if (!isVisible) return null;

    return (
        <motion.div
            className="loading-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
        >
            <div className="loading-wrapper">
                <div className="loading-content">
                    {/* Animated orbs */}
                    <div className="orb-container">
                        <motion.div
                            className="orb orb-1"
                            animate={{
                                scale: [1, 1.2, 1],
                                opacity: [0.5, 1, 0.5],
                            }}
                            transition={{
                                duration: 2,
                                repeat: Infinity,
                                ease: 'easeInOut',
                            }}
                        />
                        <motion.div
                            className="orb orb-2"
                            animate={{
                                scale: [1.2, 1, 1.2],
                                opacity: [1, 0.5, 1],
                            }}
                            transition={{
                                duration: 2,
                                repeat: Infinity,
                                ease: 'easeInOut',
                                delay: 0.5,
                            }}
                        />
                        <motion.div
                            className="orb orb-3"
                            animate={{
                                scale: [1, 1.3, 1],
                                opacity: [0.7, 1, 0.7],
                            }}
                            transition={{
                                duration: 2,
                                repeat: Infinity,
                                ease: 'easeInOut',
                                delay: 1,
                            }}
                        />
                    </div>

                    {/* Spinning ring */}
                    <motion.div
                        className="spinner-ring"
                        animate={{ rotate: 360 }}
                        transition={{
                            duration: 3,
                            repeat: Infinity,
                            ease: 'linear',
                        }}
                    />

                    {/* Text */}
                    <motion.p
                        className="loading-text gradient-text"
                        animate={{ opacity: [0.7, 1, 0.7] }}
                        transition={{
                            duration: 1.5,
                            repeat: Infinity,
                            ease: 'easeInOut',
                        }}
                    >
                        {message}
                    </motion.p>

                    {/* Particles */}
                    <div className="particles">
                        {[...Array(20)].map((_, i) => (
                            <motion.div
                                key={i}
                                className="particle"
                                style={{
                                    left: `${Math.random() * 100}%`,
                                    top: `${Math.random() * 100}%`,
                                }}
                                animate={{
                                    y: [0, -30, 0],
                                    opacity: [0, 1, 0],
                                    scale: [0, 1, 0],
                                }}
                                transition={{
                                    duration: 2 + Math.random() * 2,
                                    repeat: Infinity,
                                    delay: Math.random() * 2,
                                    ease: 'easeInOut',
                                }}
                            />
                        ))}
                    </div>
                </div>

                {/* Log Panel */}
                <motion.div
                    className="log-panel glass-card"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    <div className="log-header">
                        <span className="log-title">
                            <span className="log-icon">📋</span>
                            Activity Log
                        </span>
                        <button
                            className="log-toggle"
                            onClick={() => setShowDetails(!showDetails)}
                        >
                            {showDetails ? '▼' : '▶'}
                        </button>
                    </div>

                    <AnimatePresence>
                        {showDetails && (
                            <motion.div
                                className="log-container"
                                ref={logContainerRef}
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                            >
                                {displayLogs.map((log, i) => (
                                    <motion.div
                                        key={i}
                                        className={`log-entry log-${log.type || 'info'}`}
                                        initial={{ opacity: 0, x: -10 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: i * 0.05 }}
                                    >
                                        <span className="log-time">{log.time}</span>
                                        <span className="log-message">{log.text}</span>
                                    </motion.div>
                                ))}

                                {/* Typing indicator */}
                                <div className="log-entry log-typing">
                                    <span className="log-time">{new Date().toLocaleTimeString()}</span>
                                    <span className="log-message">
                                        <span className="typing-dots">
                                            <span>.</span><span>.</span><span>.</span>
                                        </span>
                                    </span>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    <div className="log-footer">
                        <span className="log-hint">
                            💡 First-time model download may take 2-5 minutes
                        </span>
                    </div>
                </motion.div>
            </div>
        </motion.div>
    );
}
