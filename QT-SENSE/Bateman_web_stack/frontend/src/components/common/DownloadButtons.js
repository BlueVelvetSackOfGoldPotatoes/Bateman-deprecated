// frontend/src/components/common/DownloadButtons.js

import React from 'react';
import styled from 'styled-components';
import * as XLSX from 'xlsx';
import { saveAs } from 'file-saver';

const ButtonContainer = styled.div`
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
`;

const DownloadButtons = ({ leads }) => {
  const handleDownloadCSV = () => {
    const csvContent = convertToCSV(leads);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    downloadBlob(blob, 'leads.csv');
  };

  const handleDownloadExcel = () => {
    const worksheet = XLSX.utils.json_to_sheet(leads);
    const workbook = { Sheets: { 'data': worksheet }, SheetNames: ['data'] };
    const excelBuffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
    const data = new Blob([excelBuffer], { type: 'application/octet-stream' });
    saveAs(data, 'leads.xlsx');
  };

  const convertToCSV = (objArray) => {
    if (!objArray || !objArray.length) {
      return '';
    }

    const headers = Object.keys(objArray[0]);
    const rows = objArray.map(obj => headers.map(header => {
      let cell = obj[header] ? obj[header].toString() : '';
      cell = cell.replace(/"/g, '""'); // Escape double quotes
      if (cell.search(/("|,|\n)/g) >= 0) {
        cell = `"${cell}"`;
      }
      return cell;
    }).join(','));

    return [headers.join(','), ...rows].join('\r\n');
  };

  const downloadBlob = (blob, filename) => {
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <ButtonContainer>
      <button onClick={handleDownloadCSV} style={{ padding: '0.5rem 1rem' }}>
        Download CSV
      </button>
      <button onClick={handleDownloadExcel} style={{ padding: '0.5rem 1rem' }}>
        Download Excel
      </button>
    </ButtonContainer>
  );
};

export default DownloadButtons;
