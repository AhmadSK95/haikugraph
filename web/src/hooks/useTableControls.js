import { useState, useMemo } from 'react'

export function useTableControls(data, columns, defaultPageSize = 10) {
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize, setPageSize] = useState(defaultPageSize)
  const [sortColumn, setSortColumn] = useState(null)
  const [sortDirection, setSortDirection] = useState('asc') // 'asc' or 'desc'
  const [filters, setFilters] = useState({})

  // Apply filters
  const filteredData = useMemo(() => {
    if (!data) return []
    
    return data.filter(row => {
      return Object.entries(filters).every(([column, filterValue]) => {
        if (!filterValue) return true
        const cellValue = String(row[column] ?? '').toLowerCase()
        return cellValue.includes(filterValue.toLowerCase())
      })
    })
  }, [data, filters])

  // Apply sorting
  const sortedData = useMemo(() => {
    if (!sortColumn) return filteredData

    return [...filteredData].sort((a, b) => {
      const aVal = a[sortColumn]
      const bVal = b[sortColumn]

      // Handle null/undefined
      if (aVal === null || aVal === undefined) return 1
      if (bVal === null || bVal === undefined) return -1

      // Try numeric comparison
      const aNum = Number(aVal)
      const bNum = Number(bVal)
      if (!isNaN(aNum) && !isNaN(bNum)) {
        return sortDirection === 'asc' ? aNum - bNum : bNum - aNum
      }

      // String comparison
      const aStr = String(aVal).toLowerCase()
      const bStr = String(bVal).toLowerCase()
      if (sortDirection === 'asc') {
        return aStr < bStr ? -1 : aStr > bStr ? 1 : 0
      } else {
        return bStr < aStr ? -1 : bStr > aStr ? 1 : 0
      }
    })
  }, [filteredData, sortColumn, sortDirection])

  // Paginate
  const totalPages = Math.ceil(sortedData.length / pageSize)
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * pageSize
    const end = start + pageSize
    return sortedData.slice(start, end)
  }, [sortedData, currentPage, pageSize])

  // Reset to page 1 when filters/sorting change
  const handleSort = (column) => {
    if (sortColumn === column) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc')
    } else {
      setSortColumn(column)
      setSortDirection('asc')
    }
    setCurrentPage(1)
  }

  const handleFilter = (column, value) => {
    setFilters(prev => ({
      ...prev,
      [column]: value
    }))
    setCurrentPage(1)
  }

  const handlePageChange = (newPage) => {
    setCurrentPage(Math.max(1, Math.min(newPage, totalPages)))
  }

  const handlePageSizeChange = (newSize) => {
    setPageSize(newSize)
    setCurrentPage(1)
  }

  return {
    paginatedData,
    currentPage,
    pageSize,
    totalPages,
    totalRows: sortedData.length,
    sortColumn,
    sortDirection,
    filters,
    handleSort,
    handleFilter,
    handlePageChange,
    handlePageSizeChange,
    allFilteredData: sortedData
  }
}
