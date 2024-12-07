// frontend/src/index.js

import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';

// Optionally, import global styles or reset CSS
import './index.css'; // Create this file if you want to add global styles

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);
