import React, { useState, useEffect } from 'react';
import { listDatasets, runPreprocessing, listPreprocessingRuns } from '../services/api';

/**
 * Preprocessing component managing dataset pipeline cleaning steps.
 * Dispatches run preprocessing tasks and lists recent pipeline executions.
 */
function Preprocessing() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDs, setSelectedDs] = useState(null);
  const [runs, setRuns] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    listDatasets()
      .then(res => {
        const list = res.data?.results || res.data || [];
        setDatasets(list);
        if (list.length > 0) {
          setSelectedDs(list[0].id);
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (selectedDs) {
      listPreprocessingRuns(selectedDs)
        .then(res => setRuns(res.data?.results || res.data || []))
        .catch(() => setRuns([]));
    }
  }, [selectedDs]);

  const handleRunPreprocessing = async () => {
    if (!selectedDs) return;
    setProcessing(true);
    setError('');
    setResult(null);

    try {
      const res = await runPreprocessing(selectedDs);
      setResult(res.data);
      // Refresh runs
      listPreprocessingRuns(selectedDs)
        .then(r => setRuns(r.data?.results || r.data || []))
        .catch(() => {});
    } catch (err) {
      setError(err.response?.data?.error || 'Preprocessing failed.');
    } finally {
      setProcessing(false);
    }
  };

  const StepCard = ({ step, title, icon, data, children }) => (
    <div className="card animate-in" style={{ marginBottom: 16 }}>
      <div className="card-title">{icon} Step {step}: {title}</div>
      {children}
    </div>
  );

  return (
    <div>
      <div className="page-header">
        <h2>Preprocessing Pipeline</h2>
        <p>Steps 2-6 — Duplicates, missing values, outliers, and imbalance handling</p>
      </div>

      {/* Controls */}
      <div className="card animate-in" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
            <label>Select Dataset</label>
            <select
              className="form-input"
              value={selectedDs || ''}
              onChange={(e) => setSelectedDs(Number(e.target.value))}
            >
              <option value="">-- Select --</option>
              {datasets.map(ds => (
                <option key={ds.id} value={ds.id}>{ds.name} ({ds.total_pairs} pairs)</option>
              ))}
            </select>
          </div>
          <button
            className="btn btn-primary"
            onClick={handleRunPreprocessing}
            disabled={!selectedDs || processing}
          >
            {processing ? (
              <><span className="spinner" style={{ width: 16, height: 16 }}></span> Processing...</>
            ) : '🔧 Run Steps 1-6'}
          </button>
        </div>
      </div>

      {error && (
        <div className="card" style={{ borderColor: 'var(--accent-red)', marginBottom: 24 }}>
          <p style={{ color: 'var(--accent-red)' }}>❌ {error}</p>
        </div>
      )}

      {/* Results */}
      {result?.step_reports && (
        <>
          {/* Step 2: Duplicates */}
          <StepCard step={2} title="Duplicates" icon="🔍">
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">Full Pair Duplicates</div>
                <div className="metric-value warning">
                  {result.step_reports.step_2?.duplicates_full_pair?.count || 0}
                </div>
                <div className="metric-sub">
                  {result.step_reports.step_2?.duplicates_full_pair?.percentage}%
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-label">EN-Only Duplicates</div>
                <div className="metric-value">
                  {result.step_reports.step_2?.duplicates_en_only?.count || 0}
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-label">AR-Only Duplicates</div>
                <div className="metric-value">
                  {result.step_reports.step_2?.duplicates_ar_only?.count || 0}
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-label">After Dedup</div>
                <div className="metric-value success">
                  {result.step_reports.step_2?.remaining_rows?.toLocaleString()}
                </div>
              </div>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
              {result.step_reports.step_2?.action_taken}
            </p>
          </StepCard>

          {/* Step 3: Missing Values */}
          <StepCard step={3} title="Missing Values" icon="❓">
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">EN Missing</div>
                <div className="metric-value">
                  {result.step_reports.step_3?.summary?.en_missing || 0}
                </div>
                <div className="metric-sub">{result.step_reports.step_3?.summary?.en_missing_pct}%</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">AR Missing</div>
                <div className="metric-value">
                  {result.step_reports.step_3?.summary?.ar_missing || 0}
                </div>
                <div className="metric-sub">{result.step_reports.step_3?.summary?.ar_missing_pct}%</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">After Cleaning</div>
                <div className="metric-value success">
                  {result.step_reports.step_3?.remaining_rows?.toLocaleString()}
                </div>
              </div>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
              {result.step_reports.step_3?.action_taken}
            </p>
          </StepCard>

          {/* Step 4: Outliers */}
          <StepCard step={4} title="Outliers" icon="📏">
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">Z-Score Outliers</div>
                <div className="metric-value warning">
                  {result.step_reports.step_4?.outliers?.zscore?.count || 0}
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-label">IQR Outliers</div>
                <div className="metric-value warning">
                  {result.step_reports.step_4?.outliers?.iqr?.count || 0}
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-label">After Outlier Removal</div>
                <div className="metric-value success">
                  {result.step_reports.step_4?.remaining_rows?.toLocaleString()}
                </div>
              </div>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
              {result.step_reports.step_4?.action_taken}
            </p>
          </StepCard>

          {/* Step 6: Imbalance */}
          <StepCard step={6} title="Imbalance Check" icon="⚖️">
            <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
              {result.step_reports.step_6?.note || result.step_reports.step_6?.action_taken}
            </p>
            {result.step_reports.step_6?.class_distribution && (
              <div className="metrics-grid" style={{ marginTop: 12 }}>
                {Object.entries(result.step_reports.step_6.class_distribution).map(([cls, data]) => (
                  <div className="metric-card" key={cls}>
                    <div className="metric-label">{cls}</div>
                    <div className="metric-value">{data.count}</div>
                    <div className="metric-sub">{data.percentage}%</div>
                  </div>
                ))}
              </div>
            )}
          </StepCard>

          {/* Final Summary */}
          <div className="card animate-in" style={{
            borderColor: 'var(--accent-green)',
            background: 'rgba(16, 185, 129, 0.05)',
          }}>
            <div className="card-title">✅ Preprocessing Complete</div>
            <p style={{ color: 'var(--text-secondary)', fontSize: 15 }}>
              Final cleaned dataset: <strong style={{ color: 'var(--accent-green)' }}>
                {result.final_pairs?.toLocaleString()}
              </strong> sentence pairs. Ready for training!
            </p>
          </div>
        </>
      )}

      {/* Previous Runs */}
      {runs.length > 0 && !result && (
        <div className="card animate-in">
          <div className="card-title">📋 Previous Preprocessing Runs</div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Run ID</th>
                <th>Status</th>
                <th>Original</th>
                <th>Final</th>
                <th>Duplicates</th>
                <th>Started</th>
              </tr>
            </thead>
            <tbody>
              {runs.map(run => (
                <tr key={run.id}>
                  <td>#{run.id}</td>
                  <td>
                    <span className={`badge badge-${run.status === 'completed' ? 'success' : 'warning'}`}>
                      {run.status}
                    </span>
                  </td>
                  <td>{run.original_shape?.rows?.toLocaleString()}</td>
                  <td>{run.final_pairs?.toLocaleString()}</td>
                  <td>{run.duplicate_pct}%</td>
                  <td style={{ color: 'var(--text-muted)' }}>
                    {new Date(run.started_at).toLocaleString()}
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

export default Preprocessing;
