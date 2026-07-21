import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import EDA from './pages/EDA';
import Preprocessing from './pages/Preprocessing';
import Training from './pages/Training';
import Evaluation from './pages/Evaluation';
import Translate from './pages/Translate';

/**
 * App component setting up the main router, sidebar layout, and content routing.
 */
function App() {
  return (
    <Router>
      <div className="app-container">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/eda" element={<EDA />} />
            <Route path="/preprocessing" element={<Preprocessing />} />
            <Route path="/training" element={<Training />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/translate" element={<Translate />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
