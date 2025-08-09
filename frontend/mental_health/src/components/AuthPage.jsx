import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// --- SVG Icons for Input Fields ---
const UserIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
    </svg>
);
const EmailIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M16 12a4 4 0 10-8 0 4 4 0 008 0zm0 0v1.5a2.5 2.5 0 005 0V12a9 9 0 10-9 9m4.5-1.206a8.959 8.959 0 01-4.5 1.207" />
    </svg>
);
const LockIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
    </svg>
);

// --- Main AuthPage Component ---
const AuthPage = ({ onLogin, onSignup }) => {
    const [isLoginView, setIsLoginView] = useState(true);
    const [formData, setFormData] = useState({ name: '', email: '', password: '' });
    const [error, setError] = useState(''); // State for login errors

    // --- Hardcoded user for demonstration ---
    const MOCK_USER = {
        email: 'test@user.com',
        password: 'password123',
        name: 'Alex Doe'
    };

    const handleInputChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
        setError(''); // Clear error on new input
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        setError('');

        if (isLoginView) {
            // Check credentials against the mock user
            if (formData.email === MOCK_USER.email && formData.password === MOCK_USER.password) {
                console.log("Login successful");
                onLogin(MOCK_USER); // Pass mock user data on successful login
            } else {
                setError('Invalid email or password. Please try again.');
            }
        } else {
            // For signup, we'll just simulate a success and log in the user
            console.log("Signing up with:", formData.name, formData.email, formData.password);
            // In a real app, you'd save this data, then log them in.
            onSignup(formData);
        }
    };
    
    // Animation variants remain the same
    const formVariants = { hidden: { opacity: 0, y: 50 }, visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } }};
    const nameInputVariants = { hidden: { opacity: 0, height: 0, marginTop: 0 }, visible: { opacity: 1, height: 'auto', marginTop: '1rem' }, exit: { opacity: 0, height: 0, marginTop: 0 }};

    return (
        <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-blue-100 via-purple-100 to-gray-50">
            <motion.div
                className="w-full max-w-md p-8 space-y-6 bg-white/70 backdrop-blur-lg rounded-2xl shadow-lg border border-white/30"
                variants={formVariants}
                initial="hidden"
                animate="visible"
            >
                <div className="text-center">
                    <h1 className="text-3xl font-bold text-gray-800">{isLoginView ? 'Welcome Back' : 'Create Your Account'}</h1>
                    <p className="mt-2 text-gray-500">{isLoginView ? 'Log in to continue your journey' : 'Start your path to wellness today'}</p>
                </div>

                <form className="space-y-4" onSubmit={handleSubmit}>
                    <AnimatePresence>
                        {!isLoginView && (
                            <motion.div key="name-input" variants={nameInputVariants} initial="hidden" animate="visible" exit="exit" className="relative">
                                <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"><UserIcon /></div>
                                <input type="text" name="name" placeholder="Full Name" value={formData.name} onChange={handleInputChange} required={!isLoginView} className="w-full pl-10 pr-3 py-2 bg-white/80 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-300 focus:border-purple-400 outline-none" />
                            </motion.div>
                        )}
                    </AnimatePresence>
                    
                    <div className="relative">
                         <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"><EmailIcon /></div>
                        <input type="email" name="email" placeholder="Email Address" value={formData.email} onChange={handleInputChange} required className="w-full pl-10 pr-3 py-2 bg-white/80 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-300 focus:border-purple-400 outline-none" />
                    </div>

                    <div className="relative">
                        <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"><LockIcon /></div>
                        <input type="password" name="password" placeholder="Password" value={formData.password} onChange={handleInputChange} required className="w-full pl-10 pr-3 py-2 bg-white/80 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-300 focus:border-purple-400 outline-none" />
                    </div>

                    {/* Display error message if it exists */}
                    {error && <p className="text-sm text-center text-red-500">{error}</p>}

                    <motion.button whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.98 }} type="submit" className="w-full py-3 font-semibold text-white bg-purple-500 rounded-lg hover:bg-purple-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transition-colors">
                        {isLoginView ? 'Log In' : 'Sign Up'}
                    </motion.button>
                </form>

                <div className="text-center text-sm text-gray-500">
                    <p>
                        {isLoginView ? "Don't have an account?" : "Already have an account?"}
                        <button onClick={() => setIsLoginView(!isLoginView)} className="ml-1 font-semibold text-purple-600 hover:underline">
                            {isLoginView ? 'Sign Up' : 'Log In'}
                        </button>
                    </p>
                </div>
            </motion.div>
        </div>
    );
};

export default AuthPage;
