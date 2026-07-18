import React from 'react';
import { NavLink } from 'react-router-dom';

const navItems = [
  { label: 'Overview', items: [
    { to: '/', icon: '📊', text: 'Dashboard' },
  ]},
  { label: 'Data Pipeline', items: [
    { to: '/upload', icon: '📁', text: 'Upload Dataset' },
    { to: '/preprocessing', icon: '🔧', text: 'Preprocessing' },
    { to: '/eda', icon: '📈', text: 'EDA Visualizations' },
  ]},
  { label: 'Model', items: [
    { to: '/training', icon: '🧠', text: 'Training' },
    { to: '/evaluation', icon: '🎯', text: 'Evaluation' },
  ]},
  { label: 'Demo', items: [
    { to: '/translate', icon: '🌐', text: 'Translate' },
  ]},
];

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h1>🌐 EN → AR</h1>
        <p>Machine Translator</p>
      </div>
      <nav className="sidebar-nav">
        {navItems.map((section, sIdx) => (
          <div key={sIdx}>
            <div className="sidebar-section-label">{section.label}</div>
            {section.items.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                end={item.to === '/'}
              >
                <span className="nav-icon">{item.icon}</span>
                <span>{item.text}</span>
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
      <div style={{
        padding: '16px 20px',
        borderTop: '1px solid var(--border-color)',
        fontSize: '11px',
        color: 'var(--text-muted)',
      }}>
        Graduation Project • 2024
      </div>
    </aside>
  );
}

export default Sidebar;
