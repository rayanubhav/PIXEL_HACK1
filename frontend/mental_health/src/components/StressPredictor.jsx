import React, { useState } from 'react';
import { motion } from 'framer-motion';

// --- Helper Components & Data ---
const Spinner = () => <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>;

const suggestionLibrary = {
  low: { title: "Maintain Balance", text: "You're in a great state. Try 5 minutes of mindful observation today to appreciate the calm." },
  medium: { title: "Box Breathing", text: "Calm your system. Inhale for 4s, hold for 4s, exhale for 4s, hold for 4s. Repeat 5 times." },
  high: { title: "4-7-8 Breathing", text: "A powerful calming technique. Inhale for 4s, hold your breath for 7s, and exhale slowly for 8s." }
};

// --- StressPredictor Component ---
const StressPredictor = () => {
    const [inputs, setInputs] = useState({ heart_rate: '', steps: '', sleep: '', age: '' });
    const [result, setResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleInputChange = (e) => setInputs({ ...inputs, [e.target.name]: e.target.value });

    const handleSubmit = (e) => {
        e.preventDefault();
        setIsLoading(true);
        setResult(null);

        // Simulate backend prediction
        setTimeout(() => {
            const { heart_rate, steps, sleep } = inputs;
            let score = 5;
            if (heart_rate > 85) score += 2;
            if (heart_rate < 60) score -= 1;
            if (steps < 5000) score += 1;
            if (steps > 10000) score -= 1;
            if (sleep < 6) score += 2;
            if (sleep > 8) score -= 1;
            
            score = Math.max(0, Math.min(10, Math.round(score)));
            const category = score <= 3 ? 'low' : score <= 6 ? 'medium' : 'high';

            setResult({ score, suggestion: suggestionLibrary[category] });
            setIsLoading(false);
        }, 1500);
    };

    const getGaugeColor = (level) => {
        if (level <= 3) return "text-green-500";
        if (level <= 6) return "text-yellow-500";
        return "text-red-500";
    };

    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full">
            <h3 className="text-2xl font-bold text-gray-800 mb-6">Stress Predictor</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Stress Form */}
                <div className="p-6 bg-white rounded-xl shadow-md">
                    <h4 className="font-bold text-lg text-gray-700 mb-4">Enter Your Daily Metrics</h4>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <input type="number" name="heart_rate" placeholder="Avg. Heart Rate (e.g., 75)" value={inputs.heart_rate} onChange={handleInputChange} required className="w-full p-2 border rounded-md focus:ring-2 focus:ring-purple-300 outline-none"/>
                        <input type="number" name="steps" placeholder="Daily Steps (e.g., 8000)" value={inputs.steps} onChange={handleInputChange} required className="w-full p-2 border rounded-md focus:ring-2 focus:ring-purple-300 outline-none"/>
                        <input type="number" name="sleep" placeholder="Sleep Hours (e.g., 7.5)" value={inputs.sleep} onChange={handleInputChange} required className="w-full p-2 border rounded-md focus:ring-2 focus:ring-purple-300 outline-none"/>
                        <input type="number" name="age" placeholder="Age (e.g., 35)" value={inputs.age} onChange={handleInputChange} required className="w-full p-2 border rounded-md focus:ring-2 focus:ring-purple-300 outline-none"/>
                        <button type="submit" disabled={isLoading} className="w-full py-3 font-semibold text-white bg-purple-500 rounded-lg hover:bg-purple-600 disabled:bg-purple-300 flex justify-center items-center">
                            {isLoading ? <Spinner /> : 'Predict Stress'}
                        </button>
                    </form>
                </div>
                {/* Stress Result */}
                <div className="p-6 bg-white rounded-xl shadow-md flex flex-col items-center justify-center text-center">
                    {isLoading && <Spinner />}
                    {!isLoading && !result && <p className="text-gray-500">Your results will appear here.</p>}
                    {result && (
                        <motion.div initial={{opacity: 0, scale: 0.8}} animate={{opacity: 1, scale: 1}}>
                            <p className={`text-7xl font-bold ${getGaugeColor(result.score)}`}>{result.score}<span className="text-3xl text-gray-400">/10</span></p>
                            <p className="text-gray-600 font-semibold mt-2">Your Estimated Stress Level</p>
                            <div className="mt-6 p-4 bg-green-50 rounded-lg w-full">
                                <h5 className="font-bold text-green-800">{result.suggestion.title}</h5>
                                <p className="text-sm text-green-700 mt-1">{result.suggestion.text}</p>
                            </div>
                        </motion.div>
                    )}
                </div>
            </div>
        </motion.div>
    );
};

export default StressPredictor;
