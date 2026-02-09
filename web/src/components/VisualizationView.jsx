import { useState, useRef } from 'react'
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Brush } from 'recharts'
import { Download, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, ArrowUpDown, ArrowUp, ArrowDown, Search } from 'lucide-react'
import { useTableControls } from '../hooks/useTableControls'
import { exportToCSV, exportChartAsSVG } from '../utils/exportUtils'
import './VisualizationView.css'

function DataTable({ data, columns }) {
  if (!data || data.length === 0) return <p className="no-data">No data to display</p>

  const {
    paginatedData,
    currentPage,
    pageSize,
    totalPages,
    totalRows,
    sortColumn,
    sortDirection,
    filters,
    handleSort,
    handleFilter,
    handlePageChange,
    handlePageSizeChange,
    allFilteredData
  } = useTableControls(data, columns, 10)

  const getSortIcon = (column) => {
    if (sortColumn !== column) return <ArrowUpDown size={14} />
    return sortDirection === 'asc' ? <ArrowUp size={14} /> : <ArrowDown size={14} />
  }

  const handleExport = () => {
    exportToCSV(allFilteredData, columns, 'data-export.csv')
  }

  return (
    <div className="table-container">
      <div className="table-controls">
        <div className="table-info">
          Showing {((currentPage - 1) * pageSize) + 1} to {Math.min(currentPage * pageSize, totalRows)} of {totalRows} rows
        </div>
        <button onClick={handleExport} className="export-button" title="Export as CSV">
          <Download size={16} />
          Export CSV
        </button>
      </div>

      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col}>
                  <div className="th-content">
                    <button onClick={() => handleSort(col)} className="sort-button">
                      {col}
                      {getSortIcon(col)}
                    </button>
                  </div>
                  <div className="filter-input-wrapper">
                    <Search size={12} />
                    <input
                      type="text"
                      placeholder="Filter..."
                      value={filters[col] || ''}
                      onChange={(e) => handleFilter(col, e.target.value)}
                      className="filter-input"
                    />
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paginatedData.map((row, i) => (
              <tr key={i}>
                {columns.map((col) => {
                  const value = row[col]
                  const displayValue = value === null || value === undefined 
                    ? <span className="null-value">(No data)</span>
                    : String(value)
                  return <td key={col}>{displayValue}</td>
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="table-pagination">
          <div className="pagination-controls">
            <button 
              onClick={() => handlePageChange(1)} 
              disabled={currentPage === 1}
              className="pagination-button"
              title="First page"
            >
              <ChevronsLeft size={16} />
            </button>
            <button 
              onClick={() => handlePageChange(currentPage - 1)} 
              disabled={currentPage === 1}
              className="pagination-button"
              title="Previous page"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="pagination-info">
              Page {currentPage} of {totalPages}
            </span>
            <button 
              onClick={() => handlePageChange(currentPage + 1)} 
              disabled={currentPage === totalPages}
              className="pagination-button"
              title="Next page"
            >
              <ChevronRight size={16} />
            </button>
            <button 
              onClick={() => handlePageChange(totalPages)} 
              disabled={currentPage === totalPages}
              className="pagination-button"
              title="Last page"
            >
              <ChevronsRight size={16} />
            </button>
          </div>
          <div className="page-size-selector">
            <label>Rows per page:</label>
            <select value={pageSize} onChange={(e) => handlePageSizeChange(Number(e.target.value))}>
              <option value={10}>10</option>
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>
        </div>
      )}
    </div>
  )
}

function NumberDisplay({ value, label, units }) {
  const formatValue = (val) => {
    if (val === null || val === undefined) return 'N/A'
    
    // Try to format as currency
    if (units === 'currency') {
      const numVal = Number(val)
      if (!isNaN(numVal)) {
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD'
        }).format(numVal)
      }
    }
    
    // Try to format as percentage
    if (units === 'percentage') {
      const numVal = Number(val)
      if (!isNaN(numVal)) {
        return `${numVal}%`
      }
    }
    
    // Try to format as number
    const numVal = Number(val)
    if (!isNaN(numVal)) {
      return new Intl.NumberFormat('en-US').format(numVal)
    }
    
    // For timestamps or strings, return as-is
    return String(val)
  }

  return (
    <div className="number-display">
      <span className="number-value" title={String(value)}>{formatValue(value)}</span>
      {label && <span className="number-label">{label}</span>}
    </div>
  )
}

function ChartDisplay({ data, chartType, xAxis, yAxis, units, description }) {
  if (!data || data.length === 0) return <p className="no-data">No data to display</p>

  const [hiddenSeries, setHiddenSeries] = useState(new Set())
  const chartRef = useRef(null)

  // Transform data to handle NULL values
  const transformedData = data.map(row => {
    const newRow = { ...row }
    // Replace null values with "Unknown" for display
    Object.keys(newRow).forEach(key => {
      if (newRow[key] === null || newRow[key] === undefined) {
        newRow[key] = '(No data)'
      } else if (typeof newRow[key] === 'string' && newRow[key].includes('1970-01-01')) {
        // Detect epoch time which might indicate NULL
        newRow[key] = '(No data)'
      }
    })
    return newRow
  })

  const ChartComponent = chartType === 'line' ? LineChart : BarChart
  const DataComponent = chartType === 'line' ? Line : Bar

  const handleLegendClick = (e) => {
    const dataKey = e.dataKey
    setHiddenSeries(prev => {
      const newSet = new Set(prev)
      if (newSet.has(dataKey)) {
        newSet.delete(dataKey)
      } else {
        newSet.add(dataKey)
      }
      return newSet
    })
  }

  const handleExportChart = () => {
    exportChartAsSVG(chartRef.current, `${description || 'chart'}.svg`)
  }

  const formatTooltip = (value) => {
    if (units === 'currency') {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
      }).format(value)
    }
    if (units === 'percentage') {
      return `${value}%`
    }
    return new Intl.NumberFormat('en-US').format(value)
  }

  return (
    <div className="chart-container">
      <div className="chart-controls">
        <button onClick={handleExportChart} className="export-button" title="Export as SVG">
          <Download size={16} />
          Export Chart
        </button>
      </div>
      <ResponsiveContainer width="100%" height={350} ref={chartRef}>
        <ChartComponent data={transformedData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={xAxis} />
          <YAxis />
          <Tooltip formatter={formatTooltip} />
          <Legend onClick={handleLegendClick} wrapperStyle={{ cursor: 'pointer' }} />
          {!hiddenSeries.has(yAxis) && (
            <DataComponent 
              type="monotone" 
              dataKey={yAxis} 
              fill="#667eea" 
              stroke="#667eea"
              opacity={1}
            />
          )}
          <Brush dataKey={xAxis} height={30} stroke="#667eea" />
        </ChartComponent>
      </ResponsiveContainer>
      <p className="chart-hint">💡 Use the brush at the bottom to zoom. Click legend to toggle series.</p>
    </div>
  )
}

function VisualizationView({ results }) {
  if (!results || results.length === 0) return null

  return (
    <div className="visualization-section">
      <h3>Visualizations</h3>
      
      <div className="viz-grid">
        {results.map((result, idx) => {
          const { display_hint, preview_rows, columns, chart_type, x_axis, y_axis, units, row_count } = result

          if (row_count === 0) return null

          return (
            <div key={idx} className="viz-card">
              <div className="viz-header">
                <h4>{result.description || `Result ${idx + 1}`}</h4>
                <span className="viz-badge">{row_count} rows</span>
              </div>

              {display_hint === 'number' && preview_rows.length > 0 && (
                <NumberDisplay 
                  value={Object.values(preview_rows[0])[0]}
                  label={columns[0]}
                  units={units}
                />
              )}

              {(display_hint === 'bar_chart' || display_hint === 'line_chart') && (
                <ChartDisplay
                  data={preview_rows}
                  chartType={chart_type}
                  xAxis={x_axis}
                  yAxis={y_axis}
                  units={units}
                  description={result.description}
                />
              )}

              {display_hint === 'table' && (
                <DataTable data={preview_rows} columns={columns} />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default VisualizationView
