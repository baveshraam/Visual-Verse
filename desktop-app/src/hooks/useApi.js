import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const DEFAULT_BACKEND_URL = 'http://localhost:8000';

export function useApi() {
    const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL);
    const [isConnected, setIsConnected] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [gpuInfo, setGpuInfo] = useState(null);

    // Get backend URL from Electron if available
    useEffect(() => {
        const getUrl = async () => {
            if (window.electronAPI?.getBackendUrl) {
                try {
                    const url = await window.electronAPI.getBackendUrl();
                    setBackendUrl(url);
                } catch (e) {
                    console.log('Using default backend URL');
                }
            }
        };
        getUrl();
    }, []);

    // Health check with GPU info
    const checkConnection = useCallback(async () => {
        try {
            const response = await axios.get(`${backendUrl}/health`, { timeout: 3000 });
            setIsConnected(response.status === 200);
            setError(null);

            // Capture GPU info from health response
            if (response.data?.gpu) {
                setGpuInfo(response.data.gpu);
            }

            return true;
        } catch (e) {
            setIsConnected(false);
            setError('Backend not connected');
            setGpuInfo(null);
            return false;
        }
    }, [backendUrl]);

    // Initial connection check and periodic polling
    useEffect(() => {
        checkConnection();
        const interval = setInterval(checkConnection, 10000);
        return () => clearInterval(interval);
    }, [checkConnection]);

    // Classify text
    const classifyText = useCallback(async (text) => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await axios.post(`${backendUrl}/api/classify`, {
                text
            });
            return response.data;
        } catch (e) {
            setError(e.response?.data?.detail || e.message);
            throw e;
        } finally {
            setIsLoading(false);
        }
    }, [backendUrl]);

    // Generate Comic
    const generateComic = useCallback(async (text, options = {}) => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await axios.post(`${backendUrl}/api/comic`, {
                text,
                style: options.style || 'western',
                max_panels: options.maxPanels || 4,
                use_placeholder: options.usePlaceholder ?? false
            }, {
                responseType: 'json',
                timeout: 300000 // 5 min timeout for image generation
            });
            return response.data;
        } catch (e) {
            setError(e.response?.data?.detail || e.message);
            throw e;
        } finally {
            setIsLoading(false);
        }
    }, [backendUrl]);

    // Generate Mind Map
    const generateMindMap = useCallback(async (text, options = {}) => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await axios.post(`${backendUrl}/api/mindmap`, {
                text,
                max_keywords: options.maxKeywords || 15,
                central_topic: options.centralTopic || null,
                theme: options.theme || 'dark'
            });
            return response.data;
        } catch (e) {
            setError(e.response?.data?.detail || e.message);
            throw e;
        } finally {
            setIsLoading(false);
        }
    }, [backendUrl]);

    // Process text (auto-route)
    const processText = useCallback(async (text, forceMode = null) => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await axios.post(`${backendUrl}/api/process`, {
                text,
                force_mode: forceMode
            });
            return response.data;
        } catch (e) {
            setError(e.response?.data?.detail || e.message);
            throw e;
        } finally {
            setIsLoading(false);
        }
    }, [backendUrl]);

    return {
        backendUrl,
        isConnected,
        isLoading,
        error,
        gpuInfo,
        checkConnection,
        classifyText,
        generateComic,
        generateMindMap,
        processText
    };
}

export default useApi;
