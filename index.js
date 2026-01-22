import React from 'react';
import ReactDOM from 'react-dom/client';
import ChatInterface from './ChatInterface';
import './ChatInterface.css';

const App = () => {
    return (
        <div>
            <h1>Chat with LLM</h1>
            <ChatInterface />
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);