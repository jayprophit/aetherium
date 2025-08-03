// Minimal React chat UI for Knowledge Base AI Assistant
import React, { useState } from 'react';
function ChatApp() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  async function sendMessage() {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user: 'me', message: input, history: messages.map(m => m.text) })
    });
    const data = await res.json();
    setMessages([...messages, { text: input, user: 'me' }, { text: data.response, user: 'ai' }]);
    setInput('');
  }
  return (
    <div>
      {messages.map((m, i) => <div key={i}><b>{m.user}:</b> {m.text}</div>)}
      <input value={input} onChange={e => setInput(e.target.value)} />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}
export default ChatApp;
