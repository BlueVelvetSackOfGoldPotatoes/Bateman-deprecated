// frontend/src/components/ScrapeLeadInformation/ScrapeLeadInformation.js

import React, { useState } from 'react';
import styled from 'styled-components';
import LeadTable from '../common/LeadTable';
import DownloadButtons from '../common/DownloadButtons';
import { extractLeadInformation } from '../../services/api'; // You need to implement this API function

const Container = styled.div`
  padding: 2rem;
  max-width: 800px;
  margin: 0 auto;
`;

const ScrapeLeadInformation = () => {
  const [leadsToProcess, setLeadsToProcess] = useState('');
  const [columnsToRetrieve, setColumnsToRetrieve] = useState([
    "Entity", "CEO/PI", "Researchers", "Grants",
    "Phone Number", "Email", "Country", "University",
    "Summary", "Contacts"
  ]);
  const [personKeywords, setPersonKeywords] = useState([
    "Education",
    "Current Position",
    "Expertise",
    "Email",
    "Phone Number",
    "Faculty",
    "University",
    "Bio",
    "Academic/Work Website Profile Link",
    "LinkedIn/Profile Link",
    "Facebook/Profile Link",
    "Grant",
    "Curriculum Vitae"
  ]);
  const [maxPersons, setMaxPersons] = useState(10);
  const [leadsInfo, setLeadsInfo] = useState([]);
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleScrapeLeads = async () => {
    if (!leadsToProcess.trim()) {
      setMessage('Please provide leads to process.');
      return;
    }
    if (columnsToRetrieve.length === 0) {
      setMessage('Please select at least one information field to retrieve.');
      return;
    }
    setIsLoading(true);
    setMessage('');
    try {
      // Assuming leadsToProcess is a comma-separated list of lead IDs or entities
      const leadsArray = leadsToProcess.split(',').map(lead => lead.trim()).filter(lead => lead !== '');
      const response = await extractLeadInformation(leadsArray, columnsToRetrieve, personKeywords, maxPersons);
      setLeadsInfo([...leadsInfo, ...response.data]);
      setMessage('Lead information scraped successfully!');
    } catch (error) {
      console.error(error);
      setMessage('Failed to scrape lead information.');
    }
    setIsLoading(false);
  };

  return (
    <Container>
      <h2>Analyze Lead Information</h2>
      
      {/* Leads to Process Input */}
      <div>
        <label>Leads to Process (comma-separated IDs or Entities):</label>
        <input
          type="text"
          value={leadsToProcess}
          onChange={(e) => setLeadsToProcess(e.target.value)}
          placeholder="e.g., 1,2,3 or Company A, Company B"
          style={{ width: '100%', padding: '0.5rem' }}
        />
      </div>
      
      {/* Information Fields to Retrieve */}
      <div style={{ marginTop: '1rem' }}>
        <label>Information Fields to Retrieve:</label>
        <select multiple value={columnsToRetrieve} onChange={(e) => {
          const options = e.target.options;
          const selected = [];
          for (let i = 0; i < options.length; i++) {
            if (options[i].selected) {
              selected.push(options[i].value);
            }
          }
          setColumnsToRetrieve(selected);
        }} style={{ width: '100%', padding: '0.5rem', height: '150px' }}>
          <option value="Entity">Entity</option>
          <option value="CEO/PI">CEO/PI</option>
          <option value="Researchers">Researchers</option>
          <option value="Grants">Grants</option>
          <option value="Phone Number">Phone Number</option>
          <option value="Email">Email</option>
          <option value="Country">Country</option>
          <option value="University">University</option>
          <option value="Summary">Summary</option>
          <option value="Contacts">Contacts</option>
          {/* Add dynamic fields as options */}
        </select>
      </div>
      
      {/* Person Search Keywords */}
      <div style={{ marginTop: '1rem' }}>
        <label>Person Search Keywords (comma-separated):</label>
        <input
          type="text"
          value={personKeywords.join(', ')}
          onChange={(e) => setPersonKeywords(e.target.value.split(',').map(k => k.trim()))}
          placeholder="e.g., Education, Expertise, Email"
          style={{ width: '100%', padding: '0.5rem' }}
        />
      </div>
      
      {/* Max Persons Input */}
      <div style={{ marginTop: '1rem' }}>
        <label>Maximum number of persons to process per lead:</label>
        <input
          type="number"
          value={maxPersons}
          onChange={(e) => setMaxPersons(parseInt(e.target.value))}
          min="1"
          max="100"
          style={{ width: '100%', padding: '0.5rem' }}
        />
      </div>
      
      {/* Scrape Button */}
      <button onClick={handleScrapeLeads} disabled={isLoading} style={{ padding: '0.5rem 1rem', marginTop: '1rem' }}>
        {isLoading ? 'Scraping...' : 'Search and Analyse Leads'}
      </button>
      
      {/* Display Message */}
      {message && <p style={{ marginTop: '1rem', color: 'green' }}>{message}</p>}
      
      {/* Leads Information Table */}
      {leadsInfo.length > 0 && (
        <div style={{ marginTop: '2rem' }}>
          <h3>Leads Information</h3>
          <LeadTable leads={leadsInfo} leadFields={columnsToRetrieve} personnelFields={[]} />
          <DownloadButtons leads={leadsInfo} />
        </div>
      )}
    </Container>
  );
};

export default ScrapeLeadInformation;
