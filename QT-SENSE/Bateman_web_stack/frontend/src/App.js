// frontend/src/App.js

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import styled from 'styled-components';
import InputLeads from './components/InputLeads/InputLeads';
import ScrapeLeadInformation from './components/ScrapeLeadInformation/ScrapeLeadInformation';
// Import other components as needed

const Nav = styled.nav`
  background-color: #282c34;
  padding: 1rem;
`;

const NavList = styled.ul`
  list-style: none;
  display: flex;
  gap: 1rem;
`;

const NavItem = styled.li`
  a {
    color: white;
    text-decoration: none;
    font-weight: bold;
    
    &:hover {
      text-decoration: underline;
    }
  }
`;

const App = () => {
  return (
    <Router>
      <div>
        <Nav>
          <NavList>
            <NavItem>
              <Link to="/">BATEMAN</Link>
            </NavItem>
            <NavItem>
              <Link to="/input-leads">Input Leads</Link>
            </NavItem>
            <NavItem>
              <Link to="/scrape-lead-information">Analyze Lead Information</Link>
            </NavItem>
            {/* Add other navigation links */}
          </NavList>
        </Nav>
        
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/input-leads" element={<InputLeads />} />
          <Route path="/scrape-lead-information" element={<ScrapeLeadInformation />} />
          {/* Add other routes */}
        </Routes>
      </div>
    </Router>
  );
};

const Home = () => (
  <div style={{ padding: '2rem' }}>
    <h1>Welcome to BATEMAN</h1>
    <p>Manage and analyze your leads efficiently.</p>
  </div>
);

export default App;
