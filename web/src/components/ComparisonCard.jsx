import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import './ComparisonCard.css'

function formatValue(value, isPercentage = false) {
  if (value === null || value === undefined) return 'N/A'
  if (isPercentage) return `${value.toFixed(1)}%`
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(value)
}

function ComparisonCard({ comparison }) {
  const { current_value, comparison_value, delta, delta_pct, direction } = comparison
  
  const getIcon = () => {
    if (direction === 'up') return <TrendingUp size={32} color="#10b981" />
    if (direction === 'down') return <TrendingDown size={32} color="#ef4444" />
    return <Minus size={32} color="#64748b" />
  }

  const getDirectionColor = () => {
    if (direction === 'up') return '#10b981'
    if (direction === 'down') return '#ef4444'
    return '#64748b'
  }

  return (
    <div className="comparison-card">
      <h3>Comparison</h3>
      
      <div className="comparison-grid">
        <div className="comparison-item">
          <span className="comparison-label">{comparison.current_period}</span>
          <span className="comparison-value">{formatValue(current_value)}</span>
        </div>

        <div className="comparison-direction">
          {getIcon()}
        </div>

        <div className="comparison-item">
          <span className="comparison-label">{comparison.comparison_period}</span>
          <span className="comparison-value">{formatValue(comparison_value)}</span>
        </div>
      </div>

      <div className="comparison-delta" style={{ color: getDirectionColor() }}>
        <span className="delta-amount">{formatValue(delta)}</span>
        {delta_pct !== null && (
          <span className="delta-pct">({formatValue(delta_pct, true)})</span>
        )}
      </div>
    </div>
  )
}

export default ComparisonCard
