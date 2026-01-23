// Frontend configuration loaded from backend
let config = null;

export async function loadConfig() {
    if (config) return config;
    
    try {
        const baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
        const response = await fetch(`${baseUrl}/api/config`);
        
        if (!response.ok) {
            throw new Error('Failed to load configuration from server');
        }
        
        const data = await response.json();
        config = data;
        return config;
    } catch (error) {
        console.error('Error loading config:', error);
        // Fallback configuration if backend is unavailable
        config = {
            api: {
                    exposed_host: 'localhost',
                    host: 'localhost',
                    port: 8000,
                endpoints: {
                    chat: '/chat',
                    health: '/health',
                },
            },
        };
        return config;
    }
}

export function getConfig() {
    if (!config) {
        console.warn('Config not loaded yet, using defaults');
        return {
            api: {
                    exposed_host: 'localhost',
                    host: 'localhost',
                    port: 8000,
                endpoints: {
                    chat: '/chat',
                    health: '/health',
                },
            },
        };
    }
    return config;
}
