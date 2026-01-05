import { motion } from 'framer-motion';
import './ModeSelector.css';

const modes = [
    {
        id: 'comic',
        name: 'Comic',
        icon: '🎨',
        description: 'Transform stories into visual comic strips',
        color: '#ff2d7a'
    },
    {
        id: 'mindmap',
        name: 'Mind-Map',
        icon: '🧠',
        description: 'Visualize concepts as connected nodes',
        color: '#00f5ff'
    }
];

export default function ModeSelector({ selected, onChange }) {
    return (
        <div className="mode-selector">
            <h3 className="selector-title">
                <span className="icon">⚡</span>
                Visualization Mode
            </h3>

            <div className="mode-options">
                {modes.map((mode) => (
                    <motion.button
                        key={mode.id}
                        className={`mode-option ${selected === mode.id ? 'active' : ''}`}
                        onClick={() => onChange(mode.id)}
                        whileHover={{ scale: 1.02, x: 5 }}
                        whileTap={{ scale: 0.98 }}
                        style={{
                            '--mode-color': mode.color
                        }}
                    >
                        <div className="mode-icon">{mode.icon}</div>
                        <div className="mode-info">
                            <span className="mode-name">{mode.name}</span>
                            <span className="mode-description">{mode.description}</span>
                        </div>
                        <motion.div
                            className="mode-indicator"
                            initial={false}
                            animate={{
                                scale: selected === mode.id ? 1 : 0,
                                opacity: selected === mode.id ? 1 : 0
                            }}
                        />
                    </motion.button>
                ))}
            </div>
        </div>
    );
}
