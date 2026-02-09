import { saveAs } from 'file-saver'

/**
 * Export table data as CSV
 */
export function exportToCSV(data, columns, filename = 'data.csv') {
  if (!data || data.length === 0) return

  // Create CSV header
  const header = columns.join(',')
  
  // Create CSV rows
  const rows = data.map(row => {
    return columns.map(col => {
      const value = row[col]
      // Handle values that contain commas or quotes
      if (value === null || value === undefined) return ''
      const stringValue = String(value)
      if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
        return `"${stringValue.replace(/"/g, '""')}"`
      }
      return stringValue
    }).join(',')
  })

  const csv = [header, ...rows].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  saveAs(blob, filename)
}

/**
 * Export chart as SVG
 */
export function exportChartAsSVG(chartRef, filename = 'chart.svg') {
  if (!chartRef || !chartRef.container) return

  const svgElement = chartRef.container.querySelector('svg')
  if (!svgElement) return

  const svgData = new XMLSerializer().serializeToString(svgElement)
  const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' })
  saveAs(blob, filename)
}

/**
 * Copy data to clipboard as tab-separated values
 */
export async function copyToClipboard(data, columns) {
  if (!data || data.length === 0) return false

  const header = columns.join('\t')
  const rows = data.map(row => {
    return columns.map(col => String(row[col] ?? '')).join('\t')
  })

  const text = [header, ...rows].join('\n')
  
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch (err) {
    console.error('Failed to copy to clipboard:', err)
    return false
  }
}

/**
 * Export all results as JSON
 */
export function exportToJSON(data, filename = 'data.json') {
  if (!data) return

  const json = JSON.stringify(data, null, 2)
  const blob = new Blob([json], { type: 'application/json;charset=utf-8;' })
  saveAs(blob, filename)
}
