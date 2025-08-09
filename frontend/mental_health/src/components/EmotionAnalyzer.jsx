import React, { useState } from 'react';
import { motion } from 'framer-motion';

// --- Helper Components & Data ---
const Spinner = () => <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>;

const emotionData = {
    joy: { emoji: '😊', color: 'bg-yellow-400', text: 'Joyful' },
    sadness: { emoji: '😢', color: 'bg-blue-400', text: 'Sadness' },
    anger: { emoji: '😠', color: 'bg-red-500', text: 'Anger' },
    fear: { emoji: '😨', color: 'bg-purple-400', text: 'Fear' },
    surprise: { emoji: '😮', color: 'bg-green-400', text: 'Surprise' },
    neutral: { emoji: '😐', color: 'bg-gray-400', text: 'Neutral' },
};

// --- EmotionAnalyzer Component ---
const EmotionAnalyzer = () => {
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!text.trim()) return;
        setIsLoading(true);
        setResult(null);

        // Simulate backend analysis
        setTimeout(() => {
            const lowerCaseText = text.toLowerCase();
            let detected = 'neutral';
            if (lowerCaseText.includes('happy') || lowerCaseText.includes('joy') || lowerCaseText.includes('excited')) detected = 'joy';
            else if (lowerCaseText.includes('sad') || lowerCaseText.includes('crying') || lowerCaseText.includes('depressed')) detected = 'sadness';
            else if (lowerCaseText.includes('angry') || lowerCaseText.includes('furious') || lowerCaseText.includes('hate')) detected = 'anger';
            else if (lowerCaseText.includes('scared') || lowerCaseText.includes('anxious') || lowerCaseText.includes('fear')) detected = 'fear';
            
            setResult({
                emotion: detected,
                confidence: Math.random() * (0.95 - 0.75) + 0.75, // Random confidence
            });
            setIsLoading(false);
        }, 1500);
    };

    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full">
            <h3 className="text-2xl font-bold text-gray-800 mb-6">Emotion Analyzer</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Emotion Form */}
                <div className="p-6 bg-white rounded-xl shadow-md">
                     <h4 className="font-bold text-lg text-gray-700 mb-4">How are you feeling right now?</h4>
                     <form onSubmit={handleSubmit}>
                        <textarea
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            placeholder="Write a few sentences about your current thoughts or feelings..."
                            className="w-full h-48 p-3 border rounded-md resize-none focus:ring-2 focus:ring-purple-300 outline-none"
                            required
                        />
                        <button type="submit" disabled={isLoading} className="w-full mt-4 py-3 font-semibold text-white bg-purple-500 rounded-lg hover:bg-purple-600 disabled:bg-purple-300 flex justify-center items-center">
                            {isLoading ? <Spinner /> : 'Analyze Emotion'}
                        </button>
                     </form>
                </div>
                {/* Emotion Result */}
                <div className="p-6 bg-white rounded-xl shadow-md flex flex-col items-center justify-center text-center">
                    {isLoading && <Spinner />}
                    {!isLoading && !result && <p className="text-gray-500">Your emotion analysis will appear here.</p>}
                    {result && (
                        <motion.div initial={{opacity: 0, scale: 0.8}} animate={{opacity: 1, scale: 1}}>
                            <span className="text-6xl">{emotionData[result.emotion].emoji}</span>
                            <h4 className="text-2xl font-bold text-gray-800 mt-4">{emotionData[result.emotion].text}</h4>
                            <p className="text-gray-500">Detected Emotion</p>
                            <div className="w-full bg-gray-200 rounded-full h-2.5 mt-6">
                                <div className={`${emotionData[result.emotion].color} h-2.5 rounded-full`} style={{ width: `${result.confidence * 100}%` }}></div>
                            </div>
                            <p className="text-sm font-semibold text-gray-700 mt-2">Confidence: {(result.confidence * 100).toFixed(0)}%</p>
                        </motion.div>
                    )}
                </div>
            </div>
        </motion.div>
    );
};

export default EmotionAnalyzer;
