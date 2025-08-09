/* eslint-disable no-undef */
import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios'; // Import axios for making API requests


const api = axios.create({
    baseURL: 'http://localhost:5001/api',
});

api.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
            config.headers['Authorization'] = `Bearer ${token}`;
        }
        return config;
    },
    (error) => Promise.reject(error)
);
// ===================================================================================
// --- SHARED HELPER COMPONENTS ---
// ===================================================================================
const Spinner = () => <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>;
const ButtonSpinner = () => <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>;


// ===================================================================================
// --- STRESS PREDICTOR COMPONENT ---
// ===================================================================================
const StressPredictor = () => {
    const [inputs, setInputs] = useState({ heart_rate: '', steps: '', sleep: '', age: '' });
    const [result, setResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    const handleInputChange = (e) => setInputs({ ...inputs, [e.target.name]: e.target.value });

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setResult(null);
        setError('');
        try {
            // Call the Flask backend endpoint
            const response = await axios.post('http://localhost:5001/api/predict-stress', {
                heart_rate: parseFloat(inputs.heart_rate),
                steps: parseFloat(inputs.steps),
                sleep: parseFloat(inputs.sleep),
                age: parseFloat(inputs.age)
            });
            
            const score = response.data.stress_level;
            
            const suggestionLibrary = {
              low: { title: "Maintain Balance", text: "You're in a great state. Try 5 minutes of mindful observation." },
              medium: { title: "Box Breathing", text: "Calm your system. Inhale for 4s, hold for 4s, exhale for 4s. Repeat." },
              high: { title: "4-7-8 Breathing", text: "A powerful calming technique. Inhale for 4s, hold for 7s, exhale for 8s." }
            };
            const category = score <= 3 ? 'low' : score <= 6 ? 'medium' : 'high';
            setResult({ score, suggestion: suggestionLibrary[category] });

        } catch (err) {
            setError('Could not get prediction. Please ensure the backend is running.');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
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
                <div className="p-6 bg-white rounded-xl shadow-md">
                    <h4 className="font-bold text-lg text-gray-700 mb-4">Enter Your Daily Metrics</h4>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <input type="number" name="heart_rate" placeholder="Avg. Heart Rate (e.g., 75)" value={inputs.heart_rate} onChange={handleInputChange} required className="w-full p-2 border rounded-md focus:ring-2 focus:ring-purple-300 outline-none"/>
                        <input type="number" name="steps" placeholder="Daily Steps (e.g., 8000)" value={inputs.steps} onChange={handleInputChange} required className="w-full p-2 border rounded-md focus:ring-2 focus:ring-purple-300 outline-none"/>
                        <input type="number" name="sleep" placeholder="Sleep Hours (e.g., 7.5)" value={inputs.sleep} onChange={handleInputChange} required className="w-full p-2 border rounded-md focus:ring-2 focus:ring-purple-300 outline-none"/>
                        <input type="number" name="age" placeholder="Age (e.g., 35)" value={inputs.age} onChange={handleInputChange} required className="w-full p-2 border rounded-md focus:ring-2 focus:ring-purple-300 outline-none"/>
                        <button type="submit" disabled={isLoading} className="w-full py-3 font-semibold text-white bg-purple-500 rounded-lg hover:bg-purple-600 disabled:bg-purple-300 flex justify-center items-center">
                            {isLoading ? <ButtonSpinner /> : 'Predict Stress'}
                        </button>
                    </form>
                </div>
                <div className="p-6 bg-white rounded-xl shadow-md flex flex-col items-center justify-center text-center">
                    {isLoading && <Spinner />}
                    {!isLoading && !result && !error && <p className="text-gray-500">Your results will appear here.</p>}
                    {error && <p className="text-red-500">{error}</p>}
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


// ===================================================================================
// --- EMOTION ANALYZER COMPONENT ---
// ===================================================================================
const EmotionAnalyzer = () => {
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!text.trim()) return;
        setIsLoading(true);
        setResult(null);
        setTimeout(() => {
            const lowerCaseText = text.toLowerCase();
            let detected = 'neutral';
            if (lowerCaseText.includes('happy') || lowerCaseText.includes('joy')) detected = 'joy';
            else if (lowerCaseText.includes('sad') || lowerCaseText.includes('crying')) detected = 'sadness';
            else if (lowerCaseText.includes('angry') || lowerCaseText.includes('furious')) detected = 'anger';
            else if (lowerCaseText.includes('scared') || lowerCaseText.includes('anxious')) detected = 'fear';
            const emotionData = {
                joy: { emoji: '😊', color: 'bg-yellow-400', text: 'Joyful' },
                sadness: { emoji: '😢', color: 'bg-blue-400', text: 'Sadness' },
                anger: { emoji: '😠', color: 'bg-red-500', text: 'Anger' },
                fear: { emoji: '😨', color: 'bg-purple-400', text: 'Fear' },
                surprise: { emoji: '😮', color: 'bg-green-400', text: 'Surprise' },
                neutral: { emoji: '😐', color: 'bg-gray-400', text: 'Neutral' },
            };
            setResult({ emotion: detected, confidence: Math.random() * 0.25 + 0.75, data: emotionData[detected] });
            setIsLoading(false);
        }, 1500);
    };

    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full">
            <h3 className="text-2xl font-bold text-gray-800 mb-6">Emotion Analyzer</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="p-6 bg-white rounded-xl shadow-md">
                     <h4 className="font-bold text-lg text-gray-700 mb-4">How are you feeling right now?</h4>
                     <form onSubmit={handleSubmit}>
                        <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Write about your feelings..." className="w-full h-48 p-3 border rounded-md resize-none focus:ring-2 focus:ring-purple-300 outline-none" required />
                        <button type="submit" disabled={isLoading} className="w-full mt-4 py-3 font-semibold text-white bg-purple-500 rounded-lg hover:bg-purple-600 disabled:bg-purple-300 flex justify-center items-center">
                            {isLoading ? <ButtonSpinner /> : 'Analyze Emotion'}
                        </button>
                     </form>
                </div>
                <div className="p-6 bg-white rounded-xl shadow-md flex flex-col items-center justify-center text-center">
                    {isLoading && <Spinner />}
                    {!isLoading && !result && <p className="text-gray-500">Your emotion analysis will appear here.</p>}
                    {result && (
                        <motion.div initial={{opacity: 0, scale: 0.8}} animate={{opacity: 1, scale: 1}}>
                            <span className="text-6xl">{result.data.emoji}</span>
                            <h4 className="text-2xl font-bold text-gray-800 mt-4">{result.data.text}</h4>
                            <p className="text-gray-500">Detected Emotion</p>
                            <div className="w-full bg-gray-200 rounded-full h-2.5 mt-6">
                                <div className={`${result.data.color} h-2.5 rounded-full`} style={{ width: `${result.confidence * 100}%` }}></div>
                            </div>
                            <p className="text-sm font-semibold text-gray-700 mt-2">Confidence: {(result.confidence * 100).toFixed(0)}%</p>
                        </motion.div>
                    )}
                </div>
            </div>
        </motion.div>
    );
};


// ===================================================================================
// --- CHATBOT COMPONENT ---
// ===================================================================================
const Chatbot = ({ user }) => {
    const [messages, setMessages] = useState([{ id: 'init1', text: "Hello! I'm your mindful assistant. How are you feeling today?", isUser: false }]);
    const [inputText, setInputText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const chatContainerRef = useRef(null);

    useEffect(() => { if (chatContainerRef.current) { chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight; } }, [messages]);
    
    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (inputText.trim() === '' || isLoading) return;

        const userMessage = { id: Date.now().toString(), text: inputText.trim(), isUser: true };
        setMessages(prev => [...prev, userMessage]);
        const messageText = inputText.trim();
        setInputText('');
        setIsLoading(true);

        try {
            const response = await axios.post('http://localhost:5001/api/chat', {
                message: messageText,
                user_id: user?.id || 'anonymous' // Pass user ID if available
            });
            
            const aiMessage = {
                id: (Date.now() + 1).toString(),
                text: response.data.response,
                isUser: false,
                emotion: response.data.emotion,
            };
            setMessages(prev => [...prev, aiMessage]);

        } catch (error) {
            console.error("Chat API error:", error);
            const errorMessage = {
                id: (Date.now() + 1).toString(),
                text: "Sorry, I'm having trouble connecting right now. Please try again later.",
                isUser: false,
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col h-full bg-white rounded-xl shadow-lg">
            <div className="p-4 border-b"><h3 className="text-xl font-bold text-gray-800">Mindful Chatbot</h3><p className="text-sm text-gray-500">A safe space to share.</p></div>
            <div ref={chatContainerRef} className="flex-grow p-6 overflow-y-auto"><div className="space-y-6">{messages.map(msg => (<motion.div key={msg.id} layout initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className={`flex items-end gap-3 ${msg.isUser ? 'justify-end' : 'justify-start'}`}>{!msg.isUser && <div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center mr-3 flex-shrink-0"><svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M3 9h2m14 0h2M3 15h2m14 0h2M9 6l1.646-1.646a.5.5 0 01.708 0L13 6m-4 6l1.646 1.646a.5.5 0 00.708 0L13 12" /></svg></div>}<div className={`max-w-md p-3 rounded-2xl ${msg.isUser ? 'bg-purple-500 text-white rounded-br-none' : 'bg-gray-100 text-gray-800 rounded-bl-none'}`}><p className="text-sm">{msg.text}</p></div>{msg.isUser && <div className="w-10 h-10 rounded-full bg-purple-500 text-white flex items-center justify-center ml-3 flex-shrink-0 font-bold">{user?.name?.charAt(0) || 'U'}</div>}</motion.div>))}{isLoading && <motion.div className="flex items-end gap-3 justify-start"><div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center mr-3 flex-shrink-0"><svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M3 9h2m14 0h2M3 15h2m14 0h2M9 6l1.646-1.646a.5.5 0 01.708 0L13 6m-4 6l1.646 1.646a.5.5 0 00.708 0L13 12" /></svg></div><div className="max-w-md p-3 rounded-2xl bg-gray-100"><div className="flex items-center space-x-1"><span className="h-2 w-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span><span className="h-2 w-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span><span className="h-2 w-2 bg-gray-400 rounded-full animate-bounce"></span></div></div></motion.div>}</div></div>
            <div className="p-4 border-t"><form onSubmit={handleSendMessage} className="flex items-center space-x-3"><input type="text" value={inputText} onChange={(e) => setInputText(e.target.value)} placeholder="Share what's on your mind..." disabled={isLoading} className="w-full px-4 py-2 bg-gray-100 rounded-full focus:ring-2 focus:ring-purple-300 outline-none" /><button type="submit" disabled={isLoading || !inputText.trim()}><svg xmlns="http://www.w3.org/2000/svg" className={`h-6 w-6 transition-colors ${isLoading || !inputText.trim() ? 'text-gray-400' : 'text-purple-500 hover:text-purple-600'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg></button></form></div>
        </motion.div>
    );
};


// ===================================================================================
// --- THERAPIST FINDER COMPONENT (using Leaflet) ---
// ===================================================================================
const TherapistFinder = () => {
    const [location, setLocation] = useState(null);
    const [therapists, setTherapists] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [query, setQuery] = useState('mental health therapist');
    const [activeTherapistId, setActiveTherapistId] = useState(null);
    const mapRef = useRef(null);
    const [leafletReady, setLeafletReady] = useState(false);

    const LocationIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg>;
    const CallIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" /></svg>;
    const RouteIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>;

    useEffect(() => {
        // Wait for the Leaflet script to be loaded
        const checkLeaflet = setInterval(() => {
            if (window.L) {
                setLeafletReady(true);
                clearInterval(checkLeaflet);
            }
        }, 100);

        return () => clearInterval(checkLeaflet);
    }, []);

    useEffect(() => {
        if (!leafletReady) return; // Don't do anything until Leaflet is ready

        setLoading(true);
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const { latitude, longitude } = position.coords;
                setLocation({ lat: latitude, lng: longitude });
                fetchTherapists(latitude, longitude, query);
            },
            () => {
                setError('Location access denied. Using a default location.');
                const defaultLocation = { lat: 19.2183, lng: 72.9781 }; // Thane
                setLocation(defaultLocation);
                fetchTherapists(defaultLocation.lat, defaultLocation.lng, query);
            }
        );
    }, [leafletReady]);

    const fetchTherapists = async (lat, lng, searchQuery) => {
        setLoading(true);
        setError('');
        try {
            const response = await axios.get('http://localhost:5001/api/therapists', {
                params: { lat, lng, query: searchQuery }
            });
            setTherapists(response.data || []);
            if (!response.data || response.data.length === 0) {
                setError('No therapists found nearby. Showing results from other major cities.');
            }
        } catch (err) {
            console.error("Failed to fetch therapists:", err);
            setError("Could not connect to the server. Please check if the backend is running.");
        } finally {
            setLoading(false);
        }
    };

    const handleSearch = (e) => {
        e.preventDefault();
        if (location) {
            fetchTherapists(location.lat, location.lng, query);
        }
    };

    useEffect(() => {
        if (therapists.length > 0 && leafletReady) {
            if (mapRef.current === null) {
                const map = L.map('leaflet-map').setView([therapists[0].latitude, therapists[0].longitude], 13);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                }).addTo(map);
                mapRef.current = map;
            }

            if (mapRef.current) {
                mapRef.current.eachLayer((layer) => {
                    if (layer instanceof L.Marker) {
                        mapRef.current.removeLayer(layer);
                    }
                });
                therapists.forEach(therapist => {
                    L.marker([therapist.latitude, therapist.longitude]).addTo(mapRef.current)
                        .bindPopup(`<b>${therapist.name}</b><br>${therapist.address}`);
                });
            }
        }
    }, [therapists, leafletReady]);
    
    const handleTherapistSelect = (therapist) => {
        setActiveTherapistId(therapist.id);
        if (mapRef.current && window.L) {
            mapRef.current.flyTo([therapist.latitude, therapist.longitude], 15);
        }
    };

    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex flex-col">
            <div className="mb-4">
                <h3 className="text-2xl font-bold text-gray-800">Find a Therapist Nearby</h3>
                <form onSubmit={handleSearch} className="flex items-center mt-2 bg-white p-2 rounded-lg shadow-sm border">
                    <input type="text" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search therapists or clinics..." className="w-full p-2 border-none outline-none" />
                    <button type="submit" className="px-4 py-2 bg-purple-500 text-white rounded-md hover:bg-purple-600">Search</button>
                </form>
                {error && <p className="text-orange-500 text-sm mt-2">{error}</p>}
            </div>
            <div className="flex-grow grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-[65vh]">
                <div className="lg:col-span-1 h-full overflow-y-auto pr-2">{loading ? <Spinner /> : <ul className="space-y-3">{therapists.map(therapist => (<motion.li key={therapist.id} onClick={() => handleTherapistSelect(therapist)} className={`p-4 rounded-lg cursor-pointer transition-all border-2 ${activeTherapistId === therapist.id ? 'bg-purple-100 border-purple-400 shadow-lg' : 'bg-white border-transparent hover:border-purple-200 hover:shadow-md'}`} whileHover={{ scale: 1.02 }}><h4 className="font-semibold text-gray-900">{therapist.name}</h4><p className="text-sm text-gray-600 flex items-center mt-1"><LocationIcon /> {therapist.address}</p><div className="flex items-center space-x-2 mt-3"><a href={`tel:${therapist.phone}`} className="flex items-center px-3 py-1 bg-green-500 text-white text-xs rounded-full hover:bg-green-600"><CallIcon /> Call</a><a href={`https://www.google.com/maps/dir/?api=1&destination=${therapist.latitude},${therapist.longitude}`} target="_blank" rel="noopener noreferrer" className="flex items-center px-3 py-1 bg-blue-500 text-white text-xs rounded-full hover:bg-blue-600"><RouteIcon /> Route</a></div></motion.li>))}</ul>}</div>
                <div id="leaflet-map" className="lg:col-span-2 h-full rounded-xl overflow-hidden shadow-lg relative bg-gray-200">
                    {!leafletReady && <div className="w-full h-full flex items-center justify-center"><Spinner /></div>}
                    {leafletReady && !loading && therapists.length === 0 && <div className="w-full h-full flex items-center justify-center"><p className="text-gray-500">Map will appear here once therapists are found.</p></div>}
                </div>
            </div>
        </motion.div>
    );
};

/// ===================================================================================
// --- FACIAL EMOTION DETECTOR COMPONENT ---
// ===================================================================================
const EmotionDetector = () => {
    const [status, setStatus] = useState('loading');
    const [detectedEmotion, setDetectedEmotion] = useState({ emotion: '...', confidence: 0 });
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const cameraRef = useRef(null);
    const [mediaPipeReady, setMediaPipeReady] = useState(false);

    // Helper function to calculate Euclidean distance
    const euclideanDistance = (p1, p2) => (p1 && p2) ? Math.hypot(p1.x - p2.x, p1.y - p2.y) : 0;

    // Load MediaPipe scripts dynamically
    useEffect(() => {
        const loadScript = (src, id) => new Promise((resolve, reject) => {
            if (document.getElementById(id)) { resolve(); return; }
            const script = document.createElement('script');
            script.id = id;
            script.src = src;
            script.async = true;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });

        async function initialize() {
            try {
                await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js', 'camera-utils-script');
                await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js', 'face-mesh-script');
                setMediaPipeReady(true);
            } catch (err) {
                console.error("Failed to load MediaPipe scripts:", err);
                setStatus('error');
            }
        }
        initialize();
    }, []);

    const onResults = async (results) => {
        if (!canvasRef.current || !videoRef.current) return;
        const canvasCtx = canvasRef.current.getContext('2d');
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            const landmarks = results.multiFaceLandmarks[0];
            const ear_left = (euclideanDistance(landmarks[386], landmarks[374]) + euclideanDistance(landmarks[385], landmarks[373])) / (2 * euclideanDistance(landmarks[362], landmarks[263]));
            const ear_right = (euclideanDistance(landmarks[159], landmarks[145]) + euclideanDistance(landmarks[158], landmarks[144])) / (2 * euclideanDistance(landmarks[133], landmarks[33]));
            const avg_ear = (ear_left + ear_right) / 2.0;
            const mar = euclideanDistance(landmarks[13], landmarks[14]) / euclideanDistance(landmarks[61], landmarks[291]);
            const eyebrow_dist = (euclideanDistance(landmarks[105], landmarks[10]) + euclideanDistance(landmarks[334], landmarks[10])) / 2.0;
            const jaw_drop = euclideanDistance(landmarks[175], landmarks[152]);

            try {
                const response = await axios.post('http://localhost:5001/api/predict-emotion', { avg_ear, mar, eyebrow_dist, jaw_drop });
                setDetectedEmotion(response.data);
            } catch (error) {
                console.error("Error predicting emotion:", error);
                setDetectedEmotion({ emotion: 'Error', confidence: 0 });
            }

            if (window.FaceMesh && Array.isArray(window.FaceMesh.FACEMESH_TESSELATION)) {
                for (const conn of window.FaceMesh.FACEMESH_TESSELATION) {
                    const start = landmarks[conn[0]];
                    const end = landmarks[conn[1]];
                    if (start && end) {
                        canvasCtx.beginPath();
                        canvasCtx.moveTo(start.x * canvasRef.current.width, start.y * canvasRef.current.height);
                        canvasCtx.lineTo(end.x * canvasRef.current.width, end.y * canvasRef.current.height);
                        canvasCtx.strokeStyle = 'rgba(224, 224, 224, 0.5)';
                        canvasCtx.stroke();
                    }
                }
            }
        }
        canvasCtx.restore();
    };

    useEffect(() => {
        if (!mediaPipeReady || !videoRef.current) return;

        const faceMesh = new window.FaceMesh({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}` });
        faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
        faceMesh.onResults(onResults);

        cameraRef.current = new window.Camera(videoRef.current, {
            onFrame: async () => { if (videoRef.current) { await faceMesh.send({ image: videoRef.current }); } },
            width: 640,
            height: 480,
        });
        cameraRef.current.start().then(() => setStatus('ready')).catch(() => setStatus('error'));

        return () => { if (cameraRef.current) cameraRef.current.stop(); };
    }, [mediaPipeReady]);

    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full">
            <h3 className="text-2xl font-bold text-gray-800 mb-6">Real-time Emotion Detector</h3>
            <div className="relative w-full max-w-2xl mx-auto aspect-video bg-gray-200 rounded-xl shadow-md overflow-hidden">
                {status === 'loading' && <div className="absolute inset-0 flex flex-col items-center justify-center"><Spinner /><p className="mt-2 text-gray-500">Starting camera...</p></div>}
                {status === 'error' && <div className="absolute inset-0 flex flex-col items-center justify-center"><p className="mt-2 text-red-500">Could not access camera.</p></div>}
                <video ref={videoRef} className="w-full h-full object-cover transform scaleX(-1)" autoPlay playsInline></video>
                <canvas ref={canvasRef} width="640" height="480" className="absolute top-0 left-0 w-full h-full transform scaleX(-1)"></canvas>
                <div className="absolute top-4 left-4 bg-black/50 text-white px-4 py-2 rounded-lg">
                    <p className="font-bold text-lg">Detected Emotion: <span className="text-green-300">{detectedEmotion.emotion}</span></p>
                    <p className="text-sm">Confidence: <span className="text-green-300">{detectedEmotion.confidence}%</span></p>
                </div>
            </div>
        </motion.div>
    );
};


// ===================================================================================
// --- MAIN DASHBOARD COMPONENT ---
// ===================================================================================
const Dashboard = ({ user, onLogout }) => {
    const [activeComponent, setActiveComponent] = useState('Dashboard');

    const DashboardIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" /></svg>;
    const StressIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>;
    const EmotionIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>;
    const FaceIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>;
    const ChatIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>;
    const TherapistIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg>;
    const LogoutIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" /></svg>;

    const NavItem = ({ icon, label, isActive, onClick }) => (
        <li onClick={onClick} className={`flex items-center p-3 my-2 cursor-pointer rounded-lg transition-all duration-300 ${isActive ? 'bg-purple-500 text-white shadow-lg' : 'text-gray-600 hover:bg-purple-100/50 hover:text-gray-800'}`}>
            {icon}
            <span className="ml-4 font-medium">{label}</span>
        </li>
    );

    const renderActiveComponent = () => {
        switch (activeComponent) {
            case 'Dashboard':
                return (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        <div className="p-6 bg-white rounded-xl shadow-sm hover:shadow-lg transition-shadow"><h3 className="font-semibold text-lg text-gray-700">Your Mood Today</h3><p className="text-gray-500 mt-2">Placeholder for mood tracker.</p></div>
                        <div className="p-6 bg-white rounded-xl shadow-sm hover:shadow-lg transition-shadow"><h3 className="font-semibold text-lg text-gray-700">Recent Activity</h3><p className="text-gray-500 mt-2">Placeholder for recent predictions or chats.</p></div>
                        <div className="p-6 bg-green-100 rounded-xl shadow-sm hover:shadow-lg transition-shadow"><h3 className="font-semibold text-lg text-green-800">Quick Tip</h3><p className="text-green-700 mt-2">Take 5 deep breaths to center yourself.</p></div>
                    </div>
                );
            case 'StressPredictor':
                return <StressPredictor />;
            case 'EmotionAnalyzer':
                return <EmotionAnalyzer />;
            case 'EmotionDetector': // New case for the facial emotion detector
                return <EmotionDetector />;
            case 'Chatbot':
                return <Chatbot user={user} />;
            case 'TherapistFinder':
                return <TherapistFinder />;
            default:
                return null;
        }
    };

    return (
        <div className="flex min-h-screen bg-gray-50">
            <aside className="w-64 bg-white p-6 shadow-md flex flex-col justify-between">
                <div>
                    <div className="text-center mb-10"><h1 className="text-2xl font-bold text-purple-600">MindCare</h1></div>
                    <nav>
                        <ul>
                            <NavItem icon={<DashboardIcon />} label="Dashboard" isActive={activeComponent === 'Dashboard'} onClick={() => setActiveComponent('Dashboard')} />
                            <NavItem icon={<StressIcon />} label="Stress Predictor" isActive={activeComponent === 'StressPredictor'} onClick={() => setActiveComponent('StressPredictor')} />
                            <NavItem icon={<EmotionIcon />} label="Emotion Analyzer" isActive={activeComponent === 'EmotionAnalyzer'} onClick={() => setActiveComponent('EmotionAnalyzer')} />
                            <NavItem icon={<FaceIcon />} label="Facial Emotion" isActive={activeComponent === 'EmotionDetector'} onClick={() => setActiveComponent('EmotionDetector')} />
                            <NavItem icon={<ChatIcon />} label="Mindful Chatbot" isActive={activeComponent === 'Chatbot'} onClick={() => setActiveComponent('Chatbot')} />
                            <NavItem icon={<TherapistIcon />} label="Find a Therapist" isActive={activeComponent === 'TherapistFinder'} onClick={() => setActiveComponent('TherapistFinder')} />
                        </ul>
                    </nav>
                </div>
                <div><NavItem icon={<LogoutIcon />} label="Logout" onClick={onLogout} /></div>
            </aside>
            <main className="flex-1 p-10">
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
                    {activeComponent === 'Dashboard' && (
                        <>
                            <h2 className="text-3xl font-bold text-gray-800">Welcome back, {user?.name || 'Guest'}!</h2>
                            <p className="text-gray-500 mt-1">Here's your wellness summary for today.</p>
                        </>
                    )}
                    <div className="mt-8 h-[85vh]">
                        {renderActiveComponent()}
                    </div>
                </motion.div>
            </main>
        </div>
    );
};
export default Dashboard;
