import React, { useState } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";

// --- Register Chart.js components ---
ChartJS.register(ArcElement, Tooltip, Legend);

// --- Helper Components & SVGs ---

const Spinner = () => (
  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
);

const icons = {
  mindful: (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-white/80" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M14.25 6.087c0-1.718-.669-3.3-1.846-4.478A4.952 4.952 0 009.75 0c-1.718 0-3.3.669-4.478 1.846A4.952 4.952 0 003.426 6.087c0 1.718.669 3.3 1.846 4.478A4.952 4.952 0 007.5 12.413v5.175c0 .621.504 1.125 1.125 1.125h2.25c.621 0 1.125-.504 1.125-1.125v-5.175c1.02-.38 1.846-1.343 1.846-2.648zM3.426 14.664a2.25 2.25 0 00-2.246 2.09v.002a2.25 2.25 0 002.246 2.09h1.352a2.25 2.25 0 002.25-2.25v-1.352a2.25 2.25 0 00-2.25-2.25H3.426zm14.896 0a2.25 2.25 0 012.246 2.09v.002a2.25 2.25 0 01-2.246 2.09h-1.352a2.25 2.25 0 01-2.25-2.25v-1.352a2.25 2.25 0 012.25-2.25h1.352z" /></svg>
  ),
  activity: (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-white/80" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M5.636 5.636a9 9 0 1012.728 0M12 3v9" /></svg>
  ),
  music: (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-white/80" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V7.5A2.25 2.25 0 0013.5 3h-3a2.25 2.25 0 00-2.25 2.25v1.5M9 9l-2.25 2.25M18 9l-2.25 2.25m0 0l-2.25 2.25m2.25-2.25l2.25 2.25" /></svg>
  ),
};

// --- Suggestion Data ---
const suggestionLibrary = {
  low: {
    mindful: { title: "Maintain Balance", text: "You're in a great state. Try 5 minutes of mindful observation today to appreciate the calm." },
    activity: { title: "Keep Moving", text: "Your routine is working. Maybe try a new fun activity, like dancing to your favorite song." },
    music: { title: "Upbeat Tunes", text: "Keep the positive energy flowing with some upbeat instrumental or classical music." }
  },
  medium: {
    mindful: { title: "Box Breathing", text: "Calm your system. Inhale for 4s, hold for 4s, exhale for 4s, hold for 4s. Repeat 5 times." },
    activity: { title: "Quick Walk", text: "Step away from your screen. A brisk 10-minute walk can significantly clear your head." },
    music: { title: "Calm Lo-fi Beats", text: "Put on a 'lo-fi hip hop beats to relax/study to' playlist to help you focus and de-stress." }
  },
  high: {
    mindful: { title: "4-7-8 Breathing", text: "A powerful calming technique. Inhale for 4s, hold your breath for 7s, and exhale slowly for 8s." },
    activity: { title: "Guided Meditation", text: "Find a 10-minute guided meditation for stress relief on YouTube or a wellness app." },
    music: { title: "Binaural Beats", text: "Listen with headphones to binaural beats (Theta Waves) for deep relaxation or meditation." }
  }
};

// --- Main Component ---
const WellnessForm = () => {
  const [inputs, setInputs] = useState({ heart_rate: "", steps: "", sleep: "", age: "" });
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeSuggestionIndex, setActiveSuggestionIndex] = useState(0);

  const handleChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: e.target.value });
  };

  const getSuggestions = (level) => {
    const category = level <= 3 ? 'low' : level <= 6 ? 'medium' : 'high';
    const library = suggestionLibrary[category];
    return [
        { type: 'mindful', ...library.mindful },
        { type: 'activity', ...library.activity },
        { type: 'music', ...library.music },
    ];
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);
    setResult(null);
    setActiveSuggestionIndex(0);
    try {
      const res = await axios.post("http://localhost:5000/predict", inputs);
      if (res.data.stress_level !== undefined) {
        const level = res.data.stress_level;
        const suggestions = getSuggestions(level);
        setResult({ level, suggestions });
      } else {
        setError("Received an invalid response from the server.");
      }
    } catch (err) {
      setError(err.response?.data?.error || "An error occurred. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const getGaugeColor = (level) => {
    if (level <= 3) return "text-green-500";
    if (level <= 6) return "text-yellow-500";
    return "text-red-500";
  };

  const pieData = {
    labels: ['Stressed', 'Calm'],
    datasets: [{
      data: result ? [result.level * 10, 100 - result.level * 10] : [0, 100],
      backgroundColor: ['#f87171', '#34d399'],
      borderColor: '#ffffff20',
      borderWidth: 2,
    }]
  };
  
  const pieOptions = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: { label: (context) => `${context.label}: ${context.raw}%` }
      }
    }
  };

  const handleSuggestionNav = (direction) => {
    const newIndex = activeSuggestionIndex + direction;
    if (newIndex >= 0 && newIndex < result.suggestions.length) {
        setActiveSuggestionIndex(newIndex);
    }
  };

  return (
    <div className="min-h-screen w-full bg-gray-900 flex items-center justify-center p-4 font-sans relative overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-blue-900 via-purple-900 to-gray-900 animate-gradient-xy"></div>
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 100 }}
        className="bg-white/10 backdrop-blur-xl rounded-3xl shadow-2xl p-6 sm:p-8 max-w-2xl w-full z-10 border border-white/20"
      >
        <h1 className="text-3xl font-bold text-center text-white mb-6">Wellness Insights</h1>
        <form className="space-y-4" onSubmit={handleSubmit}>
          {/* Form inputs remain the same */}
          {[
              { name: "heart_rate", placeholder: "Avg. Heart Rate (e.g., 75)" },
              { name: "steps", placeholder: "Daily Steps (e.g., 8000)" },
              { name: "sleep", placeholder: "Sleep Hours (e.g., 7.5)" },
              { name: "age", placeholder: "Age (e.g., 35)" },
          ].map((field) => (
              <motion.input whileFocus={{ scale: 1.02 }} key={field.name} name={field.name} placeholder={field.placeholder} value={inputs[field.name]} onChange={handleChange} type="number" step="any" required className="w-full bg-white/10 border border-white/30 rounded-xl px-4 py-3 text-white placeholder-gray-400 focus:ring-2 focus:ring-purple-400 outline-none transition-all duration-300" />
          ))}
          <motion.button whileTap={{ scale: 0.97 }} type="submit" disabled={isLoading} className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-purple-800 text-white font-bold py-3 rounded-xl transition-all duration-300 shadow-lg hover:shadow-purple-500/50 flex items-center justify-center">
              {isLoading ? <Spinner /> : "Predict My Stress"}
          </motion.button>
        </form>
        {error && <p className="text-red-400 text-center mt-4">{error}</p>}
        <AnimatePresence>
          {result && (
            <motion.div key="result" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mt-8">
              {/* Results Grid: Gauge and Pie Chart */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                <div className="flex flex-col items-center text-center">
                  <p className="text-white/80 font-medium">Your Stress Score</p>
                  <p className={`text-7xl font-bold ${getGaugeColor(result.level)}`}>{result.level}<span className="text-3xl text-white/50">/10</span></p>
                  <p className="text-white/70 mt-2">
                    {result.level <= 3 && "Levels are low. You're managing well."}
                    {result.level > 3 && result.level <= 6 && "Moderate stress. Time for a mindful break."}
                    {result.level > 6 && "Levels are high. Prioritize rest."}
                  </p>
                </div>
                <div className="max-w-[200px] mx-auto">
                    <Pie data={pieData} options={pieOptions} />
                </div>
              </div>
              {/* Suggestion Carousel */}
              <div className="mt-10">
                <h2 className="text-xl font-bold text-white text-center mb-4">Actionable Insights</h2>
                <div className="relative h-48">
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={activeSuggestionIndex}
                            initial={{ opacity: 0, x: 50 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -50 }}
                            transition={{ duration: 0.3 }}
                            className="bg-white/10 p-6 rounded-xl border border-white/20 w-full h-full flex flex-col justify-center items-center text-center absolute"
                        >
                            <div className="flex items-center gap-4">
                                {icons[result.suggestions[activeSuggestionIndex].type]}
                                <h3 className="text-lg font-bold text-white">{result.suggestions[activeSuggestionIndex].title}</h3>
                            </div>
                            <p className="text-white/80 mt-2">{result.suggestions[activeSuggestionIndex].text}</p>
                        </motion.div>
                    </AnimatePresence>
                </div>
                <div className="flex justify-center items-center gap-4 mt-4">
                    <button onClick={() => handleSuggestionNav(-1)} disabled={activeSuggestionIndex === 0} className="px-4 py-2 bg-white/10 rounded-full disabled:opacity-50">‹ Prev</button>
                    <div className="flex gap-2">
                        {result.suggestions.map((_, index) => (
                            <div key={index} className={`w-2 h-2 rounded-full transition-colors ${index === activeSuggestionIndex ? 'bg-white' : 'bg-white/30'}`}></div>
                        ))}
                    </div>
                    <button onClick={() => handleSuggestionNav(1)} disabled={activeSuggestionIndex === result.suggestions.length - 1} className="px-4 py-2 bg-white/10 rounded-full disabled:opacity-50">Next ›</button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
};

export default WellnessForm;
