import { useState } from 'react'
import { Send, Loader2, AlertCircle, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import VisualizationView from './components/VisualizationView'
import ExplainabilityTabs from './components/ExplainabilityTabs'
import ComparisonCard from './components/ComparisonCard'
import './App.css'

// Demo questions
const DEMO_QUESTIONS = [
  "What is total revenue?",
  "Show me revenue by customer",
  "Compare revenue this month vs last month",
  "List recent transactions",
  "How many unique customers?",
  "Show payments by date"
]

function App() {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState(null)
  const [error, setError] = useState(null)

  const askQuestion = async (q) => {
    const questionText = q || question
    if (!questionText.trim()) return

    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: questionText })
      })

      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.detail || 'Request failed')
      }

      const data = await res.json()
      setResponse(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    askQuestion()
  }

  const handleDemoClick = (q) => {
    setQuestion(q)
    askQuestion(q)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>HaikuGraph Data Assistant</h1>
        <p>Ask natural language questions about your data</p>
      </header>

      <main className="main">
        <div className="question-section">
          <form onSubmit={handleSubmit} className="question-form">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question..."
              className="question-input"
              disabled={loading}
            />
            <button type="submit" className="ask-button" disabled={loading || !question.trim()}>
              {loading ? <Loader2 className="spin" size={20} /> : <Send size={20} />}
              {loading ? 'Thinking...' : 'Ask'}
            </button>
          </form>

          <div className="demo-questions">
            <p className="demo-label">Try these examples:</p>
            <div className="demo-grid">
              {DEMO_QUESTIONS.map((q, i) => (
                <button
                  key={i}
                  onClick={() => handleDemoClick(q)}
                  className="demo-button"
                  disabled={loading}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </div>

        {error && (
          <div className="error-card">
            <AlertCircle size={24} />
            <div>
              <h3>Error</h3>
              <p>{error}</p>
              <small>Try rephrasing your question or check if the database is set up correctly.</small>
            </div>
          </div>
        )}

        {response && (
          <div className="result-section">
            {/* Final Answer */}
            <div className="answer-card">
              <h2>Answer</h2>
              <p className="answer-text">{response.final_answer}</p>
              
              {response.metadata && (
                <div className="answer-meta">
                  <span>⏱ {response.metadata.execution_time_ms}ms</span>
                  {response.metadata.total_rows > 0 && (
                    <span>📊 {response.metadata.total_rows} rows</span>
                  )}
                  {response.intent && (
                    <span>🎯 {response.intent.type}</span>
                  )}
                </div>
              )}
            </div>

            {/* Comparison Card */}
            {response.comparison && (
              <ComparisonCard comparison={response.comparison} />
            )}

            {/* Visualizations */}
            {response.results && response.results.length > 0 && (
              <VisualizationView results={response.results} />
            )}

            {/* Explainability */}
            <ExplainabilityTabs
              intent={response.intent}
              plan={response.plan}
              queries={response.queries}
              results={response.results}
              metadata={response.metadata}
              warnings={response.warnings}
            />
          </div>
        )}
      </main>
    </div>
  )
}

export default App
