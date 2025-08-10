import axios from 'axios';

// Create an instance of axios
const api = axios.create({
    baseURL: 'https://mindcare-backend-4u3f.onrender.com/api', // Your Flask backend URL
});

/* INTERCEPTOR:
  This is a powerful feature of axios. It automatically attaches the 
  authentication token (JWT) to the header of every single request 
  sent from the frontend to the backend.
*/
api.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
            config.headers['Authorization'] = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

export default api;
