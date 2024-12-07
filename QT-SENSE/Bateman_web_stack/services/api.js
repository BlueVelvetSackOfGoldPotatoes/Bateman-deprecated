// frontend/src/services/api.js

import axios from 'axios';

// Base URL of the FastAPI backend
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

// Create an Axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000, // 10 seconds timeout
});

// Optional: Add interceptors for request/response
api.interceptors.request.use(
  (config) => {
    // You can add authorization headers here if needed
    // const token = localStorage.getItem('token');
    // if (token) {
    //   config.headers['Authorization'] = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle errors globally
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

// Leads API
export const createLead = (leadData) => api.post('/leads/', leadData);
export const generateLeads = (context, num_leads, lead_types) =>
  api.post('/leads/generate', { context, num_leads, lead_types });
export const searchConferenceLeads = (conference_input, context) =>
  api.post('/leads/search-conference', { conference_input, context });
export const getLeads = (skip = 0, limit = 100) => api.get(`/leads/?skip=${skip}&limit=${limit}`);
export const updateLead = (id, leadData) => api.put(`/leads/${id}`, leadData);
export const deleteLead = (id) => api.delete(`/leads/${id}`);

// Personnel API
export const createPersonnel = (personnelData) => api.post('/personnel/', personnelData);
export const getPersonnel = (skip = 0, limit = 100) => api.get(`/personnel/?skip=${skip}&limit=${limit}`);
export const updatePersonnel = (id, personnelData) => api.put(`/personnel/${id}`, personnelData);
export const deletePersonnel = (id) => api.delete(`/personnel/${id}`);

// Additional API functions can be added here as needed

export default api;
