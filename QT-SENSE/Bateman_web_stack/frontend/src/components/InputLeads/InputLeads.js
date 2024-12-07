// frontend/src/components/InputLeads/InputLeads.js

import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import ManageFields from '../DynamicFields/ManageFields';
import LeadTable from '../common/LeadTable';
import DownloadButtons from '../common/DownloadButtons';
import { 
  createLead, 
  generateLeads, 
  searchConferenceLeads, 
  getLeads 
} from '../../services/api';

const Container = styled.div`
  padding: 2rem;
  max-width: 800px;
  margin: 0 auto;
`;

const InputLeads = () => {
  const [context, setContext] = useState('');
  const [option, setOption] = useState('Generate Leads');
  const [numLeads, setNumLeads] = useState(10);
  const [leadTypes, setLeadTypes] = useState(['Research Groups']);
  const [conferenceInput, setConferenceInput] = useState('');
  const [manualLeads, setManualLeads] = useState('');
  const [leads, setLeads] = useState([]);
  const [leadFields, setLeadFields] = useState([
    "Entity", "Category", "CEO/PI", "Country",
    "University", "Summary", "Recommendations", "Source URLs"
  ]);
  const [personnelFields, setPersonnelFields] = useState([
    "Personnel Name", "Personnel Title",
    "Personnel Email", "Personnel Phone"
  ]);
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // Fetch existing leads on component mount
  useEffect(() => {
    fetchLeads();
  }, []);
  
  const fetchLeads = async () => {
    try {
      const response = await getLeads();
      setLeads(response.data);
    } catch (error) {
      console.error(error);
      setMessage('Failed to fetch leads.');
    }
  };

  const handleGenerateLeads = async () => {
    if (!context.trim()) {
      setMessage('Please provide a context for lead generation.');
      return;
    }
    if (leadTypes.length === 0) {
      setMessage('Please specify at least one lead type.');
      return;
    }
    setIsLoading(true);
    setMessage('');
    try {
      const response = await generateLeads(context, numLeads, leadTypes);
      setLeads([...leads, ...response.data]);
      setMessage(`Generated ${response.data.length} leads successfully!`);
    } catch (error) {
      console.error(error);
      setMessage('Failed to generate leads.');
    }
    setIsLoading(false);
  };

  const handleAddManualLeads = async () => {
    if (!context.trim()) {
      setMessage('Please provide a context for lead addition.');
      return;
    }
    const lines = manualLeads.split('\n').filter(line => line.trim() !== '');
    if (lines.length === 0) {
      setMessage('No valid leads entered.');
      return;
    }
    setIsLoading(true);
    setMessage('');
    try {
      const createPromises = lines.map(line => {
        const [entity] = line.split(',');
        return createLead({ 
          type: "Company",
          entity: entity.trim(), 
          dynamic_fields: {}
        });
      });
      const results = await Promise.all(createPromises);
      setLeads([...leads, ...results.map(res => res.data)]);
      setMessage(`Added ${results.length} new leads successfully!`);
      setManualLeads('');
    } catch (error) {
      console.error(error);
      setMessage('Failed to add leads.');
    }
    setIsLoading(false);
  };

  const handleSearchConference = async () => {
    if (!context.trim()) {
      setMessage('Please provide a context for lead search.');
      return;
    }
    if (!conferenceInput.trim()) {
      setMessage('Please enter a conference name or URL.');
      return;
    }
    setIsLoading(true);
    setMessage('');
    try {
      const response = await searchConferenceLeads(conferenceInput, context);
      setLeads([...leads, ...response.data]);
      setMessage(`Added ${response.data.length} new conference leads successfully!`);
      setConferenceInput('');
    } catch (error) {
      console.error(error);
      setMessage('Failed to search conference leads.');
    }
    setIsLoading(false);
  };

  return (
    <Container>
      <h2>Input Leads</h2>
      
      {/* Context Input */}
      <div>
        <label>Context:</label>
        <textarea
          value={context}
          onChange={(e) => setContext(e.target.value)}
          placeholder="Provide context for lead generation or scraping..."
          rows="4"
          style={{ width: '100%', padding: '0.5rem' }}
        />
      </div>
      
      {/* Lead Input Options */}
      <div style={{ marginTop: '1rem' }}>
        <label>Choose how to input leads:</label>
        <select value={option} onChange={(e) => setOption(e.target.value)} style={{ width: '100%', padding: '0.5rem' }}>
          <option value="Generate Leads">Generate Leads</option>
          <option value="Add Leads Manually">Add Leads Manually</option>
          <option value="Search Leads via Conference">Search Leads via Conference</option>
        </select>
      </div>
      
      {/* Manage Dynamic Fields */}
      <ManageFields sectionType="lead" fields={leadFields} setFields={setLeadFields} />
      <ManageFields sectionType="personnel" fields={personnelFields} setFields={setPersonnelFields} />
      
      {/* Conditional Rendering Based on Option */}
      {option === "Generate Leads" && (
        <div style={{ marginTop: '1rem' }}>
          <label>Number of leads per type:</label>
          <input
            type="number"
            value={numLeads}
            onChange={(e) => setNumLeads(parseInt(e.target.value))}
            min="1"
            max="100"
            style={{ width: '100%', padding: '0.5rem' }}
          />
          
          <label>Lead Types:</label>
          <ManageFields 
            sectionType="lead" 
            fields={leadTypes} 
            setFields={setLeadTypes} 
          />
          
          <button onClick={handleGenerateLeads} disabled={isLoading} style={{ padding: '0.5rem 1rem', marginTop: '1rem' }}>
            {isLoading ? 'Generating...' : 'Generate Leads'}
          </button>
        </div>
      )}
      
      {option === "Add Leads Manually" && (
        <div style={{ marginTop: '1rem' }}>
          <label>Enter one lead per line, in the format 'Entity':</label>
          <textarea
            value={manualLeads}
            onChange={(e) => setManualLeads(e.target.value)}
            placeholder="Entity Name"
            rows="6"
            style={{ width: '100%', padding: '0.5rem' }}
          />
          <button onClick={handleAddManualLeads} disabled={isLoading} style={{ padding: '0.5rem 1rem', marginTop: '1rem' }}>
            {isLoading ? 'Adding...' : 'Add Leads'}
          </button>
        </div>
      )}
      
      {option === "Search Leads via Conference" && (
        <div style={{ marginTop: '1rem' }}>
          <label>Enter the conference name or URL:</label>
          <input
            type="text"
            value={conferenceInput}
            onChange={(e) => setConferenceInput(e.target.value)}
            placeholder="Conference Name or URL"
            style={{ width: '100%', padding: '0.5rem' }}
          />
          <button onClick={handleSearchConference} disabled={isLoading} style={{ padding: '0.5rem 1rem', marginTop: '1rem' }}>
            {isLoading ? 'Searching...' : 'Search Leads'}
          </button>
        </div>
      )}
      
      {/* Display Message */}
      {message && <p style={{ marginTop: '1rem', color: 'green' }}>{message}</p>}
      
      {/* Leads Table */}
      {leads.length > 0 && (
        <div style={{ marginTop: '2rem' }}>
          <h3>Leads</h3>
          <LeadTable leads={leads} leadFields={leadFields} personnelFields={personnelFields} />
          <DownloadButtons leads={leads} />
        </div>
      )}
    </Container>
  );
};

export default InputLeads;
