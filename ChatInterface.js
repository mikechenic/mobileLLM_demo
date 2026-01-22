import React, { useState, useEffect } from 'react';
import './ChatInterface.css';
import { loadConfig, getConfig } from './config';

const ChatInterface = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [configLoaded, setConfigLoaded] = useState(false);

    // Load configuration on component mount
    useEffect(() => {
        loadConfig().then(() => {
            setConfigLoaded(true);
            // console.log('Configuration loaded:', getConfig());
        });
    }, []);

    const handleSend = async () => {
        if (!input.trim()) return;

        // Show user message immediately
        addMessage(input, 'user');
        const userInput = input;
        setInput('');

        try {
            const currentConfig = getConfig();
            const host = currentConfig.api.host.startsWith('http')
                ? currentConfig.api.host
                : 'http://' + currentConfig.api.host;
            const apiUrl = `${host}:${currentConfig.api.port}${currentConfig.api.endpoints.chat}`;

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream',
                },
                body: JSON.stringify({ text: userInput }),
            });

            if (!response.ok || !response.body) {
                throw new Error('Failed to get streaming response from server');
            }

            // Stream reader
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            let assistantBuffer = '';
            let toolCalls = [];

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                // Parse SSE-style "data: ...\n\n" chunks
                let boundary;
                while ((boundary = buffer.indexOf('\n\n')) !== -1) {
                    const rawEvent = buffer.slice(0, boundary).trim();
                    buffer = buffer.slice(boundary + 2);
                    if (!rawEvent.startsWith('data:')) continue;
                    const payload = rawEvent.replace(/^data:\s*/, '');
                    try {
                        const evt = JSON.parse(payload);
                        if (evt.text !== undefined && evt.sender) {
                            // Accumulate assistant text
                            assistantBuffer += evt.text;
                            // Update/replace last assistant chunk to show streaming text
                            setMessages(prev => {
                                const next = [...prev];
                                // Replace last assistant/meta placeholder if last was assistant
                                if (next.length > 0 && next[next.length - 1].sender === evt.sender) {
                                    next[next.length - 1] = { text: assistantBuffer, sender: evt.sender };
                                } else {
                                    next.push({ text: assistantBuffer, sender: evt.sender });
                                }
                                return next;
                            });
                        }
                        if (evt.tool_calls) {
                            toolCalls = evt.tool_calls;
                            // Render each tool call as meta blocks
                            if (Array.isArray(toolCalls)) {
                                setMessages(prev => ([
                                    ...prev,
                                    ...toolCalls.map((tc, idx) => ({
                                        text: JSON.stringify(tc, null, 2),
                                        sender: 'tool',
                                        id: `tool-${Date.now()}-${idx}`,
                                    }))
                                ]));
                            }
                        }
                        if (evt.error) {
                            setMessages(prev => [...prev, { text: `Error: ${evt.error}`, sender: 'system' }]);
                        }
                    } catch (e) {
                        console.error('Failed to parse stream chunk', e, payload);
                    }
                }
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('Error: Could not connect to the server.', 'system');
        }
    };

    // Helper function to add messages of different types
    const addMessage = (text, sender) => {
        setMessages(prevMessages => [...prevMessages, { text, sender }]);
    };

    // Helper function to format tool message for display
    const formatToolMessage = (toolData) => {
        try {
            if (typeof toolData === 'string') {
                toolData = JSON.parse(toolData);
            }

            const { name, arguments: args, result } = toolData;
            
            // Extract the text content from result
            let resultText = '';
            if (result) {
                // Handle result as string - try to extract from TextContent pattern
                if (typeof result === 'string') {
                    // Try to extract text from TextContent pattern
                    const textMatch = result.match(/text='([^']*)'/);
                    if (textMatch && textMatch[1]) {
                        resultText = textMatch[1];
                    } else {
                        resultText = result;
                    }
                } 
                // Handle structured result with content array
                else if (result.content && Array.isArray(result.content)) {
                    resultText = result.content
                        .map(c => c.text || '')
                        .join(' ')
                        .trim();
                }
                // Handle result with text property
                else if (result.text) {
                    resultText = result.text;
                }
            }

            return { name, args, resultText };
        } catch (e) {
            console.error('Failed to parse tool message', e);
            return null;
        }
    };

    return (
        <div className="chat-container">
            <div className="messages">
                {messages.map((msg, index) => {
                    const isMeta = msg.sender === 'tool' || msg.sender === 'system';
                    
                    if (msg.sender === 'tool') {
                        const toolInfo = formatToolMessage(msg.text);
                        if (toolInfo) {
                            const { name, args, resultText } = toolInfo;
                            return (
                                <div key={index} className="meta-block tool">
                                    <div className="tool-header">
                                        <span className="tool-name">{name}</span>
                                    </div>
                                    <div className="tool-separator"></div>
                                    <div className="tool-inputs">
                                        <span className="tool-section-label">Input:</span>
                                        {args && Object.entries(args).map(([key, value]) => (
                                            <div key={key} className="tool-input-field">
                                                <span className="field-name">{key}:</span>
                                                <span className="field-value">{JSON.stringify(value)}</span>
                                            </div>
                                        ))}
                                    </div>
                                    <div className="tool-output">
                                        <span className="tool-section-label">Output:</span>
                                        <div className="tool-result-text">{resultText}</div>
                                    </div>
                                </div>
                            );
                        }
                    }
                    
                    if (isMeta) {
                        return (
                            <div key={index} className={`meta-block ${msg.sender}`}>
                                <span className="sender-label">{msg.sender}</span>
                                <div className="message-content">{msg.text}</div>
                            </div>
                        );
                    }

                    return (
                        <div key={index} className={`message ${msg.sender}`}>
                            {msg.sender !== 'user' && <span className="sender-label">{msg.sender}</span>}
                            <div className="message-content">{msg.text}</div>
                        </div>
                    );
                })}
            </div>
            <div className="input-container">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                    placeholder="Please enter your message..."
                />
                <button onClick={handleSend}>Send</button>
            </div>
        </div>
    );
};

export default ChatInterface;