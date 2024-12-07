// frontend/src/components/common/LeadTable.js

import React from 'react';
import styled from 'styled-components';

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  
  th, td {
    border: 1px solid #ddd;
    padding: 0.75rem;
    text-align: left;
  }
  
  th {
    background-color: #f2f2f2;
  }
  
  tr:nth-child(even) {
    background-color: #f9f9f9;
  }
  
  tr:hover {
    background-color: #f1f1f1;
  }
`;

const LeadTable = ({ leads, leadFields = [], personnelFields = [] }) => {
  return (
    <div>
      <Table>
        <thead>
          <tr>
            <th>ID</th>
            {leadFields.map((field, idx) => (
              <th key={`lead-${idx}`}>{field}</th>
            ))}
            {personnelFields.map((field, idx) => (
              <th key={`personnel-${idx}`}>{field}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {leads.map((lead, idx) => (
            <tr key={lead.id || idx}>
              <td>{lead.id || idx + 1}</td>
              {leadFields.map((field, index) => (
                <td key={`lead-${idx}-${index}`}>{lead[field] || 'Not Available'}</td>
              ))}
              {personnelFields.map((field, index) => (
                <td key={`personnel-${idx}-${index}`}>{lead[field] || 'Not Available'}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  );
};

export default LeadTable;
