import React, { useState, useEffect } from 'react';
import { listCharts, getChartUrl, listDatasets, getEdaReport } from '../services/api';

/**
 * EDA component rendering the exploratory data analysis charts, 
 * length distributions, and token correlation heatmaps.
 */
function EDA() {
  const [charts, setCharts] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [selectedDs, setSelectedDs] = useState(null);
  const [edaReport, setEdaReport] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [chartsRes, dsRes] = await Promise.all([
          listCharts().catch(() => ({ data: { charts: [] } })),
          listDatasets().catch(() => ({ data: [] })),
        ]);
        setCharts(chartsRes.data?.charts || []);
        const dsList = dsRes.data?.results || dsRes.data || [];
        setDatasets(dsList);
        if (dsList.length > 0) setSelectedDs(dsList[0].id);
      } catch (err) {
        console.error('EDA fetch error:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    if (selectedDs) {
      getEdaReport(selectedDs)
        .then(res => setEdaReport(res.data))
        .catch(() => setEdaReport(null));
    }
  }, [selectedDs]);

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner spinner-lg"></div>
        <p>Loading visualizations...</p>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <h2>EDA Visualizations</h2>
        <p>Step 5 — Data distribution charts and exploratory analysis</p>
      </div>

      {/* Dataset Selector */}
      {datasets.length > 0 && (
        <div className="card animate-in" style={{ marginBottom: 24 }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Select Dataset</label>
            <select
              className="form-input"
              value={selectedDs || ''}
              onChange={(e) => setSelectedDs(Number(e.target.value))}
            >
              {datasets.map(ds => (
                <option key={ds.id} value={ds.id}>{ds.name} ({ds.total_pairs} pairs)</option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Summary Stats */}
      {edaReport?.preprocessing && (
        <div className="metrics-grid animate-in">
          <div className="metric-card">
            <div className="metric-label">Original Pairs</div>
            <div className="metric-value">{edaReport.total_pairs?.toLocaleString()}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">After Cleaning</div>
            <div className="metric-value success">{edaReport.cleaned_pairs?.toLocaleString()}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Duplicates Removed</div>
            <div className="metric-value warning">{edaReport.preprocessing.duplicate_pct}%</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Outliers Removed</div>
            <div className="metric-value warning">{edaReport.preprocessing.outlier_pct}%</div>
          </div>
        </div>
      )}

      {/* Charts Grid */}
      {charts.length > 0 ? (
        <div className="charts-grid">
          {charts.map((chart, i) => (
            <div key={i} className="chart-container animate-in">
              <div className="chart-title">
                {chart.filename.replace(/chart_\d+_/, '').replace(/_/g, ' ').replace('.png', '').toUpperCase()}
              </div>
              <img
                src={getChartUrl(chart.filename, edaReport?.updated_at || Date.now())}
                alt={chart.filename}
              />
            </div>
          ))}
        </div>
      ) : (
        <div className="card" style={{ textAlign: 'center', padding: 48 }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>📊</div>
          <p style={{ color: 'var(--text-secondary)' }}>
            No charts generated yet. Run preprocessing on a dataset first.
          </p>
        </div>
      )}
    </div>
  );
}

export default EDA;
