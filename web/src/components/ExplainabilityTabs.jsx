import { useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import './ExplainabilityTabs.css'

function ExplainabilityTabs({ intent, plan, queries, results, metadata, warnings }) {
  const [activeTab, setActiveTab] = useState('intent')
  const [isExpanded, setIsExpanded] = useState(false)

  const tabs = [
    { id: 'intent', label: 'Intent' },
    { id: 'plan', label: 'Plan' },
    { id: 'sql', label: 'SQL' },
    { id: 'raw', label: 'Raw Data' },
    { id: 'metadata', label: 'Metadata' }
  ]

  return (
    <div className="explainability-section">
      <button 
        className="explainability-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span>🔍 View Details</span>
        {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
      </button>

      {isExpanded && (
        <div className="explainability-content">
          <div className="tabs">
            {tabs.map(tab => (
              <button
                key={tab.id}
                className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="tab-content">
            {activeTab === 'intent' && (
              <div className="intent-view">
                {intent ? (
                  <>
                    <div className="info-row">
                      <span className="label">Type:</span>
                      <span className="value">{intent.type}</span>
                    </div>
                    <div className="info-row">
                      <span className="label">Confidence:</span>
                      <span className="value">{(intent.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="info-row">
                      <span className="label">Rationale:</span>
                      <span className="value">{intent.rationale}</span>
                    </div>
                    <div className="info-row">
                      <span className="label">Requires Comparison:</span>
                      <span className="value">{intent.requires_comparison ? 'Yes' : 'No'}</span>
                    </div>
                  </>
                ) : (
                  <p className="empty-state">Intent classification not available</p>
                )}
              </div>
            )}

            {activeTab === 'plan' && (
              <div className="plan-view">
                <pre className="json-display">
                  {JSON.stringify(plan, null, 2)}
                </pre>
              </div>
            )}

            {activeTab === 'sql' && (
              <div className="sql-view">
                {queries && queries.length > 0 ? (
                  queries.map((sql, idx) => (
                    <div key={idx} className="sql-block">
                      <div className="sql-header">Query {idx + 1}</div>
                      <pre className="sql-code">{sql}</pre>
                    </div>
                  ))
                ) : (
                  <p className="empty-state">No SQL queries</p>
                )}
              </div>
            )}

            {activeTab === 'raw' && (
              <div className="raw-view">
                {results && results.length > 0 ? (
                  results.map((result, idx) => (
                    <div key={idx} className="raw-result-block">
                      <div className="raw-result-header">
                        <h4>{result.description || `Result ${idx + 1}`}</h4>
                        <span className="raw-badge">{result.row_count} rows</span>
                      </div>
                      <div className="raw-info">
                        <div className="info-row">
                          <span className="label">Status:</span>
                          <span className="value">{result.status}</span>
                        </div>
                        <div className="info-row">
                          <span className="label">Columns:</span>
                          <span className="value">{result.columns.join(', ')}</span>
                        </div>
                      </div>
                      <pre className="json-display">
                        {JSON.stringify(result.preview_rows, null, 2)}
                      </pre>
                    </div>
                  ))
                ) : (
                  <p className="empty-state">No raw data available</p>
                )}
              </div>
            )}

            {activeTab === 'metadata' && (
              <div className="metadata-view">
                <h4>Execution Details</h4>
                <div className="info-grid">
                  <div className="info-row">
                    <span className="label">Total Time:</span>
                    <span className="value">{metadata?.execution_time_ms}ms</span>
                  </div>
                  <div className="info-row">
                    <span className="label">Plan Time:</span>
                    <span className="value">{metadata?.plan_time_ms}ms</span>
                  </div>
                  <div className="info-row">
                    <span className="label">Execution Time:</span>
                    <span className="value">{metadata?.exec_time_ms}ms</span>
                  </div>
                  <div className="info-row">
                    <span className="label">Narration Time:</span>
                    <span className="value">{metadata?.narration_time_ms}ms</span>
                  </div>
                  <div className="info-row">
                    <span className="label">Total Rows:</span>
                    <span className="value">{metadata?.total_rows || 0}</span>
                  </div>
                  <div className="info-row">
                    <span className="label">Tables Used:</span>
                    <span className="value">{metadata?.tables_used?.join(', ') || 'N/A'}</span>
                  </div>
                </div>

                {warnings && warnings.length > 0 && (
                  <>
                    <h4>Warnings</h4>
                    <ul className="warning-list">
                      {warnings.map((warning, idx) => (
                        <li key={idx}>{warning}</li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default ExplainabilityTabs
