import React, { useState, useEffect, useRef } from 'react';

interface Message {
  role: 'assistant' | 'user';
  content: string;
  time: string;
  toolEvent?: string;
  citations?: string[];
}

export const App: React.FC = () => {
  // Determine dynamic time-of-day greeting
  const [greeting, setGreeting] = useState('Good morning');
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content:
        "Hi, I'm Kishan's portfolio assistant. Feel free to ask about his work experience, skills, projects, or anything from his resume.",
      time: '4:42 AM',
    },
  ]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeTool, setActiveTool] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Dynamic greeting based on current time
    const hour = new Date().getHours();
    if (hour < 12) {
      setGreeting('Good morning');
    } else if (hour < 18) {
      setGreeting('Good afternoon');
    } else {
      setGreeting('Good evening');
    }

    // Initialize or restore session ID from localStorage
    let sId = localStorage.getItem('kishan_portfolio_session');
    if (!sId) {
      sId = 'recruiter_' + Math.random().toString(36).substring(2, 11);
      localStorage.setItem('kishan_portfolio_session', sId);
    }
    setSessionId(sId);
  }, []);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: isStreaming ? 'auto' : 'smooth' });
    }
  }, [messages, activeTool, isStreaming]);

  const getCurrentTimeFormatted = () => {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const handleSendMessage = async (customText?: string) => {
    const textToSend = (customText || input).trim();
    if (!textToSend || isStreaming) return;

    setInput('');
    const timeNow = getCurrentTimeFormatted();

    // Append user message
    setMessages((prev) => [
      ...prev,
      {
        role: 'user',
        content: textToSend,
        time: timeNow,
      },
      {
        role: 'assistant',
        content: '',
        time: timeNow,
      },
    ]);

    setIsStreaming(true);
    setActiveTool(null);

    try {
      const response = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: textToSend,
        }),
      });

      if (!response.body) {
        throw new Error('No response stream received from backend.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let accumulatedReply = '';

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
              try {
                const toolData = JSON.parse(nextLine.slice(6));
                setActiveTool(toolData.tool || 'Calling tool...');
              } catch {
                setActiveTool('Executing agent tool...');
              }
            }
          } else if (line.startsWith('event: tool_end')) {
            setActiveTool(null);
          } else if (line.startsWith('data: ')) {
            const rawData = line.slice(6).trim();
            if (!rawData) continue;
            try {
              const data = JSON.parse(rawData);
              if (data.delta) {
                accumulatedReply += data.delta;
                const currentText = accumulatedReply;
                setMessages((prev) => {
                  const lastIndex = prev.length - 1;
                  if (lastIndex < 0) return prev;
                  const last = prev[lastIndex];
                  if (last.role !== 'assistant') return prev;
                  return [
                    ...prev.slice(0, lastIndex),
                    { ...last, content: currentText },
                  ];
                });
              }
            } catch {
              // Ignore malformed chunks
            }
          }
        }
      }
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last && last.role === 'assistant') {
          last.content =
            "I'm currently unable to connect to the backend server. Please verify the FastAPI service is running.";
        }
        return updated;
      });
    } finally {
      setIsStreaming(false);
      setActiveTool(null);
    }
  };

  const suggestedQuestions = [
    { label: 'Show me recent work', icon: '✦', query: 'Show me recent work and portfolio projects' },
    { label: 'Briefly tell me about your work experience', icon: '💡', query: 'Briefly tell me about your work experience' },
    { label: "What's been on your mind lately?", icon: '💬', query: 'Explain Robin’s focus in AI/ML and backend systems' },
    { label: 'Explain the ATS project', icon: '⚡', query: 'Explain the ATS/Resume Screening project and its architecture' },
  ];

  const renderFormattedContent = (content: string) => {
    if (!content) return null;
    // Automatically strip robotic opening prefixes if generated
    const cleaned = content
      .replace(/^According to Robin['’]?s verified knowledge,?\s*:?/i, '')
      .replace(/^Based on Robin['’]?s verified knowledge,?\s*:?/i, '')
      .trim();

    const lines = cleaned.split('\n');
    return lines.map((line, lineIdx) => {
      // Parse markdown bold **text** into clean HTML <strong> without raw asterisks
      const parts = line.split(/(\*\*[^*]+\*\*)/g);
      const elements = parts.map((part, pIdx) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={pIdx}>{part.slice(2, -2)}</strong>;
        }
        return part;
      });

      return (
        <React.Fragment key={lineIdx}>
          {elements}
          {lineIdx < lines.length - 1 && <br />}
        </React.Fragment>
      );
    });
  };

  return (
    <div className="portfolio-layout">
      {/* Left Column: Editorial Bio */}
      <aside className="editorial-col">
        <div className="editorial-content">
          <h1 className="editorial-headline">{greeting}</h1>
          <p className="editorial-intro">
            Let's chat. I built this AI version of my portfolio for fun. Ask it about my work
            or what I like to do, etc.
          </p>
          <p className="editorial-bio">
            I am a designer and technologist, having always sat at the intersection of both. I
            love to design, and I also love to build.
          </p>
        </div>

        <div className="editorial-footer">
          <a
            href="https://github.com/alexsharma-dev"
            target="_blank"
            rel="noopener noreferrer"
            className="github-link"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
            </svg>
            <span>GitHub</span>
          </a>
          <button
            className="sub-link"
            onClick={() => handleSendMessage('Tell me about the ATS/Resume Screening project')}
          >
            Looking for Project Ariadne?
          </button>
        </div>
      </aside>

      {/* Right Column: Chat Stream & Top Bar */}
      <main className="chat-col">
        <header className="top-bar">
          <span className="top-tag">KISHAN / SOFTWARE DEVELOPMENT & AI</span>
        </header>

        <div className="messages-scroll">
          {messages.map((m, index) => (
            <div key={index} className={`msg-block ${m.role}`}>
              <div className="msg-text">
                {m.content ? (
                  renderFormattedContent(m.content)
                ) : isStreaming && index === messages.length - 1 ? (
                  <span className="thinking-badge">
                    <span className="pulse-dot"></span>
                    <span>Thinking...</span>
                  </span>
                ) : null}
              </div>
              <div className="msg-meta">{m.time}</div>

              {/* Show suggested pills only below initial assistant greeting when conversation has not started */}
              {index === 0 && messages.length <= 1 && (
                <div className="suggested-pills-wrap">
                  {suggestedQuestions.map((q, qIdx) => (
                    <button
                      key={qIdx}
                      className="pill-btn"
                      onClick={() => handleSendMessage(q.query)}
                      disabled={isStreaming}
                    >
                      <span className="pill-icon">{q.icon}</span>
                      <span>{q.label}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}

          {activeTool && (
            <div className="tool-status-badge">
              <span className="pulse-dot"></span>
              <span>Running: {activeTool}</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Pinned Bottom Input Form */}
        <div className="chat-bottom-wrapper">
          <form
            className="chat-input-form"
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage();
            }}
          >
            <input
              type="text"
              className="chat-input-field"
              placeholder="Ask something..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isStreaming}
              autoFocus
            />
            <button
              type="submit"
              className="submit-arrow-btn"
              disabled={isStreaming || !input.trim()}
              title="Submit message"
            >
              →
            </button>
          </form>
        </div>
      </main>
    </div>
  );
};
