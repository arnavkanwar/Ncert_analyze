import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Query NCERT for relevant paragraphs
export const queryNCERT = async (question) => {
  try {
    const response = await apiClient.post('/query', {
      question: question,
    });
    return response.data;
  } catch (error) {
    console.error('Error querying NCERT:', error);
    throw new Error(error.response?.data?.detail || error.message || 'Failed to query NCERT database');
  }
};

export const fetchPyqList = async (limit = 300) => {
  try {
    const response = await apiClient.get('/pyqs', {
      params: { limit },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching PYQ list:', error);
    throw new Error(error.response?.data?.detail || error.message || 'Failed to fetch PYQ list');
  }
};

export const queryByPyqId = async (pyqId) => {
  try {
    const response = await apiClient.post('/query/by-pyq', {
      pyq_id: pyqId,
    });
    return response.data;
  } catch (error) {
    console.error('Error querying by pyq_id:', error);
    throw new Error(error.response?.data?.detail || error.message || 'Failed to query selected PYQ');
  }
};

export default apiClient;

