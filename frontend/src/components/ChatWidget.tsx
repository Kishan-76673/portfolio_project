import React, { useState, useEffect, useRef } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  toolEvent?: string;
}

export const ChatWidget: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: "Hello! I am Alex Sharma's AI Career Assistant. Ask me anything about Alex's engineering experience, tech stack, projects, or live system verifications."
    }
  ]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeTool, setActiveTool] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize persistent session ID
  useEffect(() => {
    let sId = localStorage.getItem('portfolio_chat_session');
    if (!sId) {
      sId = 'recruiter_' + Math.random().toString(36).substring(2, 11);
      localStorage.setItem('portfolio_chat_session', sId);
    }
    setSessionId(sId);
  }, []);

  // Auto-scroll to bottom of conversation
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, activeTool]);

  const handleSend = async (messageText?: string) => {
    const textToSend = messageText || input;
    if (!textToSend.trim() || isStreaming) return;

    setInput('');
    const newMessages: Message[] = [...messages, { role: 'user', content: textToSend }];
    setMessages(newMessages);
    setIsStreaming(true);
    setActiveTool(null);

    // Placeholder for assistant stream
    setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);

    try {
      const response = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: textToSend
        })
      });

      if (!response.body) throw new Error('No response body stream');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (let i = 0; i < lines.length; i++) {
          const line = lines[i];
          if (line.startsWith('event: tool_start')) {
            const nextLine = lines[i + 1];
            if (nextLine && nextLine.startsWith('data: ')) {
              const data = JSON.parse(nextLine.slice(6));
              setActiveTool(`Executing Tool: ${data.tool}`);
            }
          } else if (line.startsWith('event: tool_end')) {
            setActiveTool(null);
          } else if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (!dataStr) continue;
            try {
              const data = JSON.parse(dataStr);
              if (data.delta) {
                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last && last.role === 'assistant') {
                    last.content += data.delta;
                  }
                  return updated;
                });
              }
            } catch {
              // Ignore malformed chunks
            }
          }
        }
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Connection error. Please check if the backend server is running.' }
      ]);
    } finally {
      setIsStreaming(false);
      setActiveTool(null);
    }
  };

  const sampleQuestions = [
    "What are Alex's core technical skills?",
    "Tell me about the AI Portfolio project",
    "Is his live demo running right now?",
    "What was his impact at DataMatrix Labs?"
  ];

  return (
    <div className="chat-shell">
      <div className="chat-messages">
        {messages.map((m, idx) => (
          <div key={idx} className={`message-row ${m.role}`}>
            <div className={`avatar ${m.role === 'user' ? 'user-avatar' : 'ai-avatar'}`}>
              {m.role === 'user' ? 'R' : 'AI'}
            </div>
            <div>
              <div className="bubble">{m.content}</div>
            </div>
          </div>
        ))}

        {activeTool && (
          <div className="tool-indicator">
            <span className="pulse-dot"></span>
            {activeTool}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div style={{ padding: '0 1.25rem 0.5rem' }}>
        <div className="suggestions">
          {sampleQuestions.map((q, i) => (
            <button
              key={i}
              className="suggestion-chip"
              onClick={() => handleSend(q)}
              disabled={isStreaming}
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      <form
        className="chat-input-bar"
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
      >
        <input
          type="text"
          className="chat-input"
          placeholder="Ask a question about Alex's background or skills..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isStreaming}
        />
        <button type="submit" className="send-button" disabled={isStreaming || !input.trim()}>
          {isStreaming ? 'Thinking...' : 'Send'}
        </button>
      </form>
    </div>
  );
};
