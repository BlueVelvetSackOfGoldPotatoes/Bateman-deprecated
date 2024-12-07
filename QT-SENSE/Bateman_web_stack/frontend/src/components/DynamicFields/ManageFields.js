// frontend/src/components/DynamicFields/ManageFields.js

import React, { useState } from 'react';
import styled from 'styled-components';

const Container = styled.div`
  margin-bottom: 2rem;
`;

const ManageFields = ({ sectionType, fields, setFields }) => {
  const [newField, setNewField] = useState('');
  const [fieldToRemove, setFieldToRemove] = useState('');

  const handleAddField = () => {
    const trimmedField = newField.trim();
    if (!trimmedField) {
      alert(`Field name cannot be empty for ${sectionType}.`);
      return;
    }
    if (fields.includes(trimmedField)) {
      alert(`Field "${trimmedField}" already exists.`);
      return;
    }
    setFields([...fields, trimmedField]);
    setNewField('');
  };

  const handleRemoveField = () => {
    if (!fieldToRemove) {
      alert(`Please select a field to remove from ${sectionType}.`);
      return;
    }
    setFields(fields.filter((field) => field !== fieldToRemove));
    setFieldToRemove('');
  };

  return (
    <Container>
      <h3>Manage {sectionType === 'lead' ? 'Lead' : 'Personnel'} Fields</h3>
      
      {/* Add Field */}
      <div>
        <input
          type="text"
          placeholder={`Add new ${sectionType === 'lead' ? 'Lead' : 'Personnel'} field`}
          value={newField}
          onChange={(e) => setNewField(e.target.value)}
          style={{ padding: '0.5rem', width: '70%', marginRight: '1rem' }}
        />
        <button onClick={handleAddField} style={{ padding: '0.5rem 1rem' }}>
          Add Field
        </button>
      </div>
      
      {/* Remove Field */}
      <div style={{ marginTop: '1rem' }}>
        <select
          value={fieldToRemove}
          onChange={(e) => setFieldToRemove(e.target.value)}
          style={{ padding: '0.5rem', width: '70%', marginRight: '1rem' }}
        >
          <option value="">Select field to remove</option>
          {fields.map((field, idx) => (
            <option key={idx} value={field}>{field}</option>
          ))}
        </select>
        <button onClick={handleRemoveField} style={{ padding: '0.5rem 1rem' }}>
          Remove Field
        </button>
      </div>
    </Container>
  );
};

export default ManageFields;
