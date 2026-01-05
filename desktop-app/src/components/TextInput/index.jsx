import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './TextInput.css';

const SAMPLE_TEXTS = {
    story: `Once upon a time, in a small village nestled between two mountains, there lived a young girl named Maya. She had always dreamed of exploring the world beyond the peaks.

One sunny morning, Maya discovered a hidden path behind the old oak tree. "This must lead somewhere magical!" she exclaimed with excitement.

She followed the path deeper into the woods. The trees grew taller and the sunlight filtered through the leaves creating dancing shadows.

Suddenly, she arrived at a clearing. In the center stood an ancient stone tower covered in ivy. A mysterious light glowed from the highest window. Maya knew her adventure was just beginning.`,

    info: `Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn from and make predictions based on data.

It includes three main types: supervised learning uses labeled data to train models, unsupervised learning discovers hidden patterns in unlabeled data, and reinforcement learning uses rewards to guide decision-making.

Deep learning, a specialized form of machine learning, uses neural networks with multiple layers to model complex patterns. Applications include natural language processing, computer vision, and speech recognition.

Therefore, machine learning has become essential for modern AI systems, enabling everything from recommendation engines to autonomous vehicles.`,

    // Tamil Story Sample
    tamilStory: `ஒரு சிறிய கிராமத்தில் மாயா என்ற பெண் வாழ்ந்தாள். அவள் மலைகளைத் தாண்டி உலகை ஆராய கனவு கண்டாள்.

ஒரு நாள் காலையில், பழைய ஆலமரத்தின் பின்னால் ஒரு மறைவான பாதையை கண்டுபிடித்தாள். "இது ஏதோ மந்திர இடத்திற்கு செல்ல வேண்டும்!" என்று மகிழ்ச்சியுடன் கூறினாள்.

அவள் காட்டின் ஆழத்திற்குள் நடந்தாள். மரங்கள் உயரமாக வளர்ந்தன, சூரிய ஒளி இலைகள் வழியாக நுழைந்து நிழல்களை உருவாக்கியது.

திடீரென்று, ஒரு வெட்டவெளியை அடைந்தாள். மையத்தில் ஒரு பழமையான கோபுரம் நின்றது. உச்சியில் இருந்து மர்மமான ஒளி வந்தது. மாயாவின் சாகசம் இப்போதுதான் தொடங்குகிறது.`,

    // Tamil Educational Sample
    tamilInfo: `இயந்திர கற்றல் என்பது செயற்கை நுண்ணறிவின் ஒரு பகுதியாகும். இது தரவுகளிலிருந்து கற்று கணிப்புகளை செய்யும் வழிமுறைகளை உருவாக்குவதில் கவனம் செலுத்துகிறது.

மூன்று முக்கிய வகைகள் உள்ளன: மேற்பார்வையிடப்பட்ட கற்றல், மேற்பார்வையிடப்படாத கற்றல், மற்றும் வலுவூட்டல் கற்றல்.

ஆழ்ந்த கற்றல் என்பது இயந்திர கற்றலின் சிறப்பு வடிவமாகும். இது பல அடுக்கு நரம்பியல் வலைப்பின்னல்களைப் பயன்படுத்துகிறது.

எனவே, இயந்திர கற்றல் நவீன AI அமைப்புகளுக்கு இன்றியமையாததாகிவிட்டது, பரிந்துரை இயந்திரங்கள் முதல் தன்னியக்க வாகனங்கள் வரை எல்லாவற்றையும் இயக்குகிறது.`
};

export default function TextInput({ value, onChange, onGenerate, isLoading }) {
    const textareaRef = useRef(null);
    const [charCount, setCharCount] = useState(0);
    const [isFocused, setIsFocused] = useState(false);

    useEffect(() => {
        setCharCount(value?.length || 0);
    }, [value]);

    const handleTextChange = (e) => {
        onChange(e.target.value);
    };

    const loadSample = (type) => {
        onChange(SAMPLE_TEXTS[type]);
        textareaRef.current?.focus();
    };

    const isValid = charCount >= 50;

    return (
        <div className="text-input-container">
            <div className="text-input-header">
                <h3 className="section-title">
                    <span className="icon">📝</span>
                    Input Text
                    <span className="language-badge">EN / தமிழ்</span>
                </h3>
                <div className="sample-buttons">
                    <motion.button
                        className="sample-btn"
                        onClick={() => loadSample('story')}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                    >
                        <span className="sample-icon">📖</span>
                        Story
                    </motion.button>
                    <motion.button
                        className="sample-btn"
                        onClick={() => loadSample('info')}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                    >
                        <span className="sample-icon">📚</span>
                        Info
                    </motion.button>
                    <motion.button
                        className="sample-btn tamil"
                        onClick={() => loadSample('tamilStory')}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                    >
                        <span className="sample-icon">📖</span>
                        தமிழ் கதை
                    </motion.button>
                    <motion.button
                        className="sample-btn tamil"
                        onClick={() => loadSample('tamilInfo')}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                    >
                        <span className="sample-icon">📚</span>
                        தமிழ் தகவல்
                    </motion.button>
                </div>
            </div>

            <div className={`textarea-wrapper ${isFocused ? 'focused' : ''} ${!isValid && charCount > 0 ? 'invalid' : ''}`}>
                <textarea
                    ref={textareaRef}
                    className="text-input"
                    value={value}
                    onChange={handleTextChange}
                    onFocus={() => setIsFocused(true)}
                    onBlur={() => setIsFocused(false)}
                    placeholder="Enter your text here... (minimum 50 characters)

Paste a story for Comic generation, or informational content for Mind-Map visualization."
                    disabled={isLoading}
                />

                <div className="textarea-footer">
                    <span className={`char-count ${!isValid ? 'invalid' : 'valid'}`}>
                        {charCount} / 50 min
                    </span>
                    <AnimatePresence>
                        {!isValid && charCount > 0 && (
                            <motion.span
                                className="validation-hint"
                                initial={{ opacity: 0, x: 10 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 10 }}
                            >
                                Need {50 - charCount} more characters
                            </motion.span>
                        )}
                    </AnimatePresence>
                </div>
            </div>

            <motion.button
                className="generate-btn"
                onClick={onGenerate}
                disabled={!isValid || isLoading}
                whileHover={isValid && !isLoading ? { scale: 1.02, y: -2 } : {}}
                whileTap={isValid && !isLoading ? { scale: 0.98 } : {}}
            >
                <AnimatePresence mode="wait">
                    {isLoading ? (
                        <motion.span
                            key="loading"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="loading-state"
                        >
                            <span className="spinner"></span>
                            Processing...
                        </motion.span>
                    ) : (
                        <motion.span
                            key="generate"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                        >
                            <span className="btn-icon">🚀</span>
                            Generate Visualization
                        </motion.span>
                    )}
                </AnimatePresence>
            </motion.button>
        </div>
    );
}
