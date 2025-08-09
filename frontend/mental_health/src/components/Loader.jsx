import React, { useEffect } from 'react';
import { motion } from 'framer-motion';

// A simple SVG logo that captures the "mind" theme.
const MindCareLogo = () => (
  <svg width="80" height="80" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M50 90C72.0914 90 90 72.0914 90 50C90 27.9086 72.0914 10 50 10C27.9086 10 10 27.9086 10 50C10 61.2741 14.5359 71.5134 22 78.9998" stroke="rgba(255,255,255,0.6)" strokeWidth="4" strokeLinecap="round"/>
    <path d="M35 50C35 41.7157 41.7157 35 50 35C58.2843 35 65 41.7157 65 50C65 58.2843 58.2843 65 50 65" stroke="rgba(255,255,255,0.8)" strokeWidth="4" strokeLinecap="round"/>
    <path d="M50 25C54.1421 25 57.5 28.3579 57.5 32.5" stroke="#a7f3d0" strokeWidth="4" strokeLinecap="round"/>
  </svg>
);


// The main Loader component
// FIX: Added a default empty function for onLoadingComplete to prevent crashes.
const Loader = ({ onLoadingComplete = () => {} }) => {

  // This effect will notify the parent component that the loading animation is done.
  useEffect(() => {
    const timer = setTimeout(() => {
      onLoadingComplete();
    }, 4000); // Animation duration is roughly 4 seconds

    return () => clearTimeout(timer);
  }, [onLoadingComplete]);

  // Animation variants for the container to orchestrate child animations
  const containerVariants = {
    start: {
      transition: {
        staggerChildren: 0.2,
      },
    },
    end: {
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  // Animation for the central logo and text
  const logoVariants = {
    start: { opacity: 0, scale: 0.8 },
    end: { opacity: 1, scale: 1, transition: { duration: 1 } },
  };
  
  // Base animation for the expanding orbs
  const orbVariants = {
    start: { scale: 0, opacity: 0 },
    end: { 
        scale: [0, 1.2, 1], 
        opacity: [0, 0.7, 1, 0], 
        transition: { duration: 2.5, times: [0, 0.3, 0.6, 1] } 
    },
  };

  return (
    <motion.div
      className="relative flex items-center justify-center w-full min-h-screen bg-gradient-to-br from-[#0a192f] via-[#1c2a4a] to-[#2a3a64] overflow-hidden"
      variants={containerVariants}
      initial="start"
      animate="end"
    >
      {/* Central Content: Logo and App Name */}
      <motion.div
        className="z-10 flex flex-col items-center"
        variants={logoVariants}
      >
        <MindCareLogo />
        <h1 className="mt-4 text-4xl font-bold text-white/90 tracking-wider">
          MindCare
        </h1>
      </motion.div>

      {/* Animated Orbs */}
      {/* Each orb has a different color and delay for a staggered, organic effect */}
      <motion.div
        className="absolute w-48 h-48 bg-blue-400/50 rounded-full"
        variants={orbVariants}
        transition={{ delay: 0.5 }}
      />
      <motion.div
        className="absolute w-32 h-32 bg-green-300/50 rounded-full"
        variants={orbVariants}
        transition={{ delay: 0.8 }}
      />
      <motion.div
        className="absolute w-40 h-40 bg-purple-400/50 rounded-full"
        variants={orbVariants}
        transition={{ delay: 1.1 }}
      />
       <motion.div
        className="absolute w-24 h-24 bg-pink-300/50 rounded-full"
        variants={orbVariants}
        transition={{ delay: 1.4 }}
      />
    </motion.div>
  );
};

export default Loader;
