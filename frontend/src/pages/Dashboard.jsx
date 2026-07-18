import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { listDatasets, listTrainingJobs } from '../services/api';

function Dashboard() {
  const [datasets, setDatasets] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [dsRes, jobsRes] = await Promise.all([
          listDatasets().catch(() => ({ data: [] })),
          listTrainingJobs().catch(() => ({ data: [] })),
        ]);
        setDatasets(dsRes.data?.results || dsRes.data || []);
        setJobs(jobsRes.data?.results || jobsRes.data || []);
      } catch (err) {
        console.error('Dashboard fetch error:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const latestDataset = datasets[0];
  const latestJob = jobs[0];
  const totalPairs = datasets.reduce((sum, d) => sum + (d.total_pairs || 0), 0);
  const cleanedPairs = datasets.reduce((sum, d) => sum + (d.cleaned_pairs || 0), 0);

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner spinner-lg"></div>
        <p>Loading dashboard...</p>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <h2>Dashboard</h2>
        <p>Project overview — English to Arabic Machine Translator</p>
      </div>

      {/* Metrics Summary */}
      <div className="metrics-grid">
        <div className="metric-card animate-in">
          <div className="metric-label">Datasets</div>
          <div className="metric-value info">{datasets.length}</div>
          <div className="metric-sub">Uploaded corpora</div>
        </div>
        <div className="metric-card animate-in">
          <div className="metric-label">Total Pairs</div>
          <div className="metric-value">{totalPairs.toLocaleString()}</div>
          <div className="metric-sub">EN-AR sentence pairs</div>
        </div>
        <div className="metric-card animate-in">
          <div className="metric-label">Cleaned Pairs</div>
          <div className="metric-value success">{cleanedPairs.toLocaleString()}</div>
          <div className="metric-sub">After preprocessing</div>
        </div>
        <div className="metric-card animate-in">
          <div className="metric-label">Training Jobs</div>
          <div className="metric-value info">{jobs.length}</div>
          <div className="metric-sub">{jobs.filter(j => j.status === 'completed').length} completed</div>
        </div>
        <div className="metric-card animate-in">
          <div className="metric-label">Best BLEU</div>
          <div className={`metric-value ${latestJob?.best_val_bleu >= 25 ? 'success' : 'warning'}`}>
            {latestJob?.best_val_bleu?.toFixed(2) || '—'}
          </div>
          <div className="metric-sub">Target: ≥ 25.0</div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card animate-in" style={{ marginBottom: 24 }}>
        <div className="card-title">🚀 Quick Actions</div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <Link to="/upload" className="btn btn-primary">📁 Upload Dataset</Link>
          <Link to="/translate" className="btn btn-success">🌐 Try Translator</Link>
          {latestDataset && (
            <Link to="/preprocessing" className="btn btn-secondary">🔧 Run Preprocessing</Link>
          )}
        </div>
      </div>

      {/* Workflow Steps */}
      <div className="card animate-in">
        <div className="card-title">📋 8-Step Workflow</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 12 }}>
          {[
            { step: 1, title: 'Load & Explore', icon: '📂', page: '/upload' },
            { step: 2, title: 'Handle Duplicates', icon: '🔍', page: '/preprocessing' },
            { step: 3, title: 'Missing Values', icon: '❓', page: '/preprocessing' },
            { step: 4, title: 'Check Outliers', icon: '📏', page: '/preprocessing' },
            { step: 5, title: 'Visualizations', icon: '📊', page: '/eda' },
            { step: 6, title: 'Check Imbalance', icon: '⚖️', page: '/preprocessing' },
            { step: 7, title: 'Train Model', icon: '🧠', page: '/training' },
            { step: 8, title: 'Evaluate', icon: '🎯', page: '/evaluation' },
          ].map(({ step, title, icon, page }) => (
            <Link key={step} to={page} style={{ textDecoration: 'none' }}>
              <div className="metric-card" style={{ cursor: 'pointer' }}>
                <div style={{ fontSize: 24, marginBottom: 8 }}>{icon}</div>
                <div className="metric-label">Step {step}</div>
                <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>{title}</div>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Recent Datasets */}
      {datasets.length > 0 && (
        <div className="card animate-in" style={{ marginTop: 24 }}>
          <div className="card-title">📂 Recent Datasets</div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Total Pairs</th>
                <th>Cleaned</th>
                <th>Status</th>
                <th>Uploaded</th>
              </tr>
            </thead>
            <tbody>
              {datasets.slice(0, 5).map((ds) => (
                <tr key={ds.id}>
                  <td style={{ fontWeight: 600 }}>{ds.name}</td>
                  <td>{ds.total_pairs?.toLocaleString()}</td>
                  <td>{ds.cleaned_pairs?.toLocaleString()}</td>
                  <td>
                    <span className={`badge badge-${
                      ds.status === 'completed' ? 'success' :
                      ds.status === 'training' ? 'running' :
                      ds.status === 'ready' ? 'info' : 'warning'
                    }`}>
                      {ds.status}
                    </span>
                  </td>
                  <td style={{ color: 'var(--text-muted)' }}>
                    {new Date(ds.uploaded_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
