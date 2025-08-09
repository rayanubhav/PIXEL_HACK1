import React, { useEffect, useState, useRef } from 'react';
import { motion } from 'framer-motion';

// --- Helper Components & SVGs ---
const Spinner = () => (
    <div className="flex justify-center items-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>
    </div>
);

const LocationIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg>;
const CallIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" /></svg>;
const RouteIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>;


// --- Mock Data (as a fallback until backend is connected) ---
const mockTherapists = [
    { id: 'ChIJ0a2v_b7P5zsR4n2F-J2sAgQ', name: 'Dr. Anjali Mehta, Mind & Soul Clinic', address: 'Gokhale Road, Thane West', lat: 19.206, lng: 72.972, phone: '+911234567890' },
    { id: 'ChIJVVVVVVVV5zsR2y-m-J2sAgQ', name: 'Serene Pathways Counseling', address: 'Panch Pakhadi, Thane West', lat: 19.192, lng: 72.978, phone: '+911234567891' },
    { id: 'ChIJ_x-m-J2sAgQRVVVVVVVV5zs', name: 'Hope & Healing Psychotherapy', address: 'Vasant Vihar, Thane West', lat: 19.225, lng: 72.965, phone: '+911234567892' },
];

// --- Google Maps API Script Loader Hook ---
const useGoogleMapsScript = (apiKey) => {
    const [isLoaded, setIsLoaded] = useState(false);
    useEffect(() => {
        if (window.google && window.google.maps) {
            setIsLoaded(true);
            return;
        }
        const script = document.createElement('script');
        script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=places`;
        script.async = true;
        script.defer = true;
        script.onload = () => setIsLoaded(true);
        document.head.appendChild(script);
        return () => { document.head.removeChild(script); };
    }, [apiKey]);
    return isLoaded;
};

// --- Main TherapistFinder Component ---
const TherapistFinder = () => {
    const GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"; // IMPORTANT: Replace with your actual key
    const mapsLoaded = useGoogleMapsScript(GOOGLE_MAPS_API_KEY);
    
    const [location, setLocation] = useState(null);
    const [therapists, setTherapists] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [activeTherapistId, setActiveTherapistId] = useState(null);

    const mapRef = useRef(null);
    const mapContainerRef = useRef(null);
    const markersRef = useRef({});

    // 1. Get User's Location
    useEffect(() => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const { latitude, longitude } = position.coords;
                    setLocation({ lat: latitude, lng: longitude });
                    fetchTherapists(latitude, longitude);
                },
                () => {
                    setError('Location access denied. Showing results for a default area.');
                    // Fallback to a default location if permission is denied
                    const defaultLocation = { lat: 19.2183, lng: 72.9781 }; // Thane
                    setLocation(defaultLocation);
                    fetchTherapists(defaultLocation.lat, defaultLocation.lng);
                }
            );
        } else {
            setError("Geolocation is not supported by this browser.");
            setLoading(false);
        }
    }, []);

    // 2. Fetch Therapist Data (using mock data for now)
    const fetchTherapists = async (lat, lng, query = 'therapist') => {
        setLoading(true);
        // --- REAL API CALL (Commented out for now) ---
        /*
        try {
            const res = await axios.get(`http://YOUR_BACKEND_URL/api/therapists?lat=${lat}&lng=${lng}&query=${query}`);
            setTherapists(res.data ?? []);
        } catch (err) {
            console.error('API Error:', err);
            setError('Could not fetch therapists. Please try again later.');
            setTherapists(mockTherapists); // Use mock data as fallback on API error
        } finally {
            setLoading(false);
        }
        */
        // --- MOCK DATA IMPLEMENTATION ---
        setTimeout(() => {
            setTherapists(mockTherapists);
            setLoading(false);
        }, 1000); // Simulate network delay
    };

    // 3. Initialize Map and Markers
    useEffect(() => {
        if (mapsLoaded && location && mapContainerRef.current && therapists.length > 0) {
            const map = new window.google.maps.Map(mapContainerRef.current, {
                center: location,
                zoom: 14,
                disableDefaultUI: true,
                styles: [ { stylers: [{ "saturation": -100 }, { "lightness": 10 }] } ] // Simplified calm style
            });
            mapRef.current = map;

            therapists.forEach(therapist => {
                const marker = new window.google.maps.Marker({
                    position: { lat: therapist.lat, lng: therapist.lng },
                    map: map,
                    title: therapist.name,
                });
                marker.addListener('click', () => handleTherapistSelect(therapist));
                markersRef.current[therapist.id] = marker;
            });
        }
    }, [mapsLoaded, location, therapists]);

    // 4. Handle User Interaction
    const handleTherapistSelect = (therapist) => {
        setActiveTherapistId(therapist.id);
        if (mapRef.current) {
            mapRef.current.panTo({ lat: therapist.lat, lng: therapist.lng });
            mapRef.current.setZoom(15);
        }
    };

    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex flex-col">
            <div className="mb-6">
                <h3 className="text-2xl font-bold text-gray-800">Find a Therapist Nearby</h3>
                {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
            </div>
            
            <div className="flex-grow grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-[65vh]">
                {/* --- Therapists List --- */}
                <div className="lg:col-span-1 h-full overflow-y-auto pr-2 custom-scrollbar">
                    {loading ? <Spinner /> : (
                        <ul className="space-y-3">
                            {therapists.map(therapist => (
                                <motion.li
                                    key={therapist.id}
                                    onClick={() => handleTherapistSelect(therapist)}
                                    className={`p-4 rounded-lg cursor-pointer transition-all border-2 ${activeTherapistId === therapist.id ? 'bg-purple-100 border-purple-400 shadow-lg' : 'bg-white border-transparent hover:border-purple-200 hover:shadow-md'}`}
                                    whileHover={{ scale: 1.02 }}
                                >
                                    <h4 className="font-semibold text-gray-900">{therapist.name}</h4>
                                    <p className="text-sm text-gray-600 flex items-center mt-1"><LocationIcon /> {therapist.address}</p>
                                    <div className="flex items-center space-x-2 mt-3">
                                        <a href={`tel:${therapist.phone}`} className="flex items-center px-3 py-1 bg-green-500 text-white text-xs rounded-full hover:bg-green-600 transition-colors"><CallIcon /> Call</a>
                                        <a href={`https://www.google.com/maps/dir/?api=1&destination=${therapist.lat},${therapist.lng}`} target="_blank" rel="noopener noreferrer" className="flex items-center px-3 py-1 bg-blue-500 text-white text-xs rounded-full hover:bg-blue-600 transition-colors"><RouteIcon /> Route</a>
                                    </div>
                                </motion.li>
                            ))}
                        </ul>
                    )}
                </div>

                {/* --- Map Container --- */}
                <div className="lg:col-span-2 h-full rounded-xl overflow-hidden shadow-lg relative bg-gray-200">
                    {mapsLoaded ? (
                        <div ref={mapContainerRef} className="w-full h-full" />
                    ) : (
                        <div className="w-full h-full flex items-center justify-center"><p className="text-gray-500">Loading Map...</p></div>
                    )}
                    {GOOGLE_MAPS_API_KEY === "YOUR_GOOGLE_MAPS_API_KEY" && (
                        <div className="absolute top-0 left-0 w-full p-2 bg-yellow-200 text-center text-yellow-800 text-xs font-semibold">
                            Please replace "YOUR_GOOGLE_MAPS_API_KEY" to display the map.
                        </div>
                    )}
                </div>
            </div>
        </motion.div>
    );
};

export default TherapistFinder;
