import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// --- SVG Icons ---
const SendIcon = ({ isDisabled }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={`h-6 w-6 transition-colors ${isDisabled ? 'text-gray-400' : 'text-purple-500 hover:text-purple-600'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
    </svg>
);
const BotIcon = () => (
    <div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center mr-3 flex-shrink-0">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M3 9h2m14 0h2M3 15h2m14 0h2M9 6l1.646-1.646a.5.5 0 01.708 0L13 6m-4 6l1.646 1.646a.5.5 0 00.708 0L13 12" />
        </svg>
    </div>
);
const UserIcon = ({ user }) => (
    <div className="w-10 h-10 rounded-full bg-purple-500 text-white flex items-center justify-center ml-3 flex-shrink-0 font-bold">
        {user?.name?.charAt(0) || 'U'}
    </div>
);


// --- Main Chatbot Component ---
const Chatbot = ({ user }) => {
    const [messages, setMessages] = useState([
        {
            id: 'init1',
            text: "Hello! I'm your mindful assistant. I'm here to listen without judgment. How are you feeling today?",
            isUser: false,
            timestamp: new Date(),
        },
    ]);
    const [inputText, setInputText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const chatContainerRef = useRef(null);

    // Auto-scroll to the bottom when new messages are added
    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [messages]);

    // Mock AI response logic
    const getAIResponse = (userMessage) => {
        const responses = [
            "Thank you for sharing that with me. It's brave to open up.",
            "I hear you. It sounds like you're going through a lot right now.",
            "That sounds really challenging. Can you tell me a bit more about what's on your mind?",
            "It takes courage to express these feelings. Remember to be kind to yourself.",
            "I understand. Let's just sit with that for a moment. No need to rush.",
        ];
        // In a real app, you would analyze userMessage here.
        return responses[Math.floor(Math.random() * responses.length)];
    };
    
    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (inputText.trim() === '' || isLoading) return;

        const userMessage = {
            id: Date.now().toString(),
            text: inputText.trim(),
            isUser: true,
            timestamp: new Date(),
        };

        setMessages(prev => [...prev, userMessage]);
        const messageText = inputText.trim();
        setInputText('');
        setIsLoading(true);

        // Simulate backend call and AI response
        setTimeout(() => {
            const aiResponseText = getAIResponse(messageText);
            const aiMessage = {
                id: (Date.now() + 1).toString(),
                text: aiResponseText,
                isUser: false,
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, aiMessage]);
            setIsLoading(false);
        }, 1500 + Math.random() * 500); // Simulate network and thinking delay
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col h-full bg-white rounded-xl shadow-lg"
        >
            {/* Header */}
            <div className="p-4 border-b border-gray-200">
                <h3 className="text-xl font-bold text-gray-800">Mindful Chatbot</h3>
                <p className="text-sm text-gray-500">A safe space to share your thoughts.</p>
            </div>

            {/* Messages Area */}
            <div ref={chatContainerRef} className="flex-grow p-6 overflow-y-auto custom-scrollbar">
                <div className="space-y-6">
                    <AnimatePresence>
                        {messages.map(msg => (
                            <motion.div
                                key={msg.id}
                                layout
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0 }}
                                className={`flex items-end gap-3 ${msg.isUser ? 'justify-end' : 'justify-start'}`}
                            >
                                {!msg.isUser && <BotIcon />}
                                <div className={`max-w-md p-3 rounded-2xl ${msg.isUser ? 'bg-purple-500 text-white rounded-br-none' : 'bg-gray-100 text-gray-800 rounded-bl-none'}`}>
                                    <p className="text-sm">{msg.text}</p>
                                </div>
                                {msg.isUser && <UserIcon user={user} />}
                            </motion.div>
                        ))}
                    </AnimatePresence>
                    {isLoading && (
                         <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-end gap-3 justify-start">
                             <BotIcon />
                             <div className="max-w-md p-3 rounded-2xl bg-gray-100 text-gray-800 rounded-bl-none">
                                 <div className="flex items-center space-x-1">
                                     <span className="h-2 w-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                                     <span className="h-2 w-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                                     <span className="h-2 w-2 bg-gray-400 rounded-full animate-bounce"></span>
                                 </div>
                             </div>
                         </motion.div>
                    )}
                </div>
            </div>

            {/* Input Form */}
            <div className="p-4 border-t border-gray-200 bg-white rounded-b-xl">
                <form onSubmit={handleSendMessage} className="flex items-center space-x-3">
                    <input
                        type="text"
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        placeholder="Share what's on your mind..."
                        disabled={isLoading}
                        className="w-full px-4 py-2 bg-gray-100 border-transparent rounded-full focus:ring-2 focus:ring-purple-300 focus:outline-none transition"
                    />
                    <button type="submit" disabled={isLoading || !inputText.trim()}>
                        <SendIcon isDisabled={isLoading || !inputText.trim()} />
                    </button>
                </form>
            </div>
        </motion.div>
    );
};

export default Chatbot;
