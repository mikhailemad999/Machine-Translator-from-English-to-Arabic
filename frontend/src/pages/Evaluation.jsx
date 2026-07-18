import React, { useState, useEffect } from 'react';
import { listTrainingJobs, getEvaluation, runEvaluation, getChartUrl } from '../services/api';

function Evaluation() {
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listTrainingJobs()
      .then((res) => {
        const completedJobs = (res.data?.results || res.data || []).filter(
          (j) => j.status === 'completed'
        );
        setJobs(completedJobs);
        if (completedJobs.length > 0) {
          setSelectedJobId(completedJobs[0].id);
        }
      })
      .catch((err) => console.error('Error fetching jobs:', err))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (selectedJobId) {
      setError('');
      setEvaluation(null);
      getEvaluation(selectedJobId)
        .then((res) => {
          setEvaluation(res.data);
        })
        .catch((err) => {
          // If not evaluated yet, we just keep evaluation null
          console.log('No evaluation found yet for this job.');
        });
    }
  }, [selectedJobId]);

  const handleRunEvaluation = async () => {
    if (!selectedJobId) return;
    setEvaluating(true);
    setError('');
    setEvaluation(null);

    try {
      const res = await runEvaluation(selectedJobId);
      setEvaluation(res.data?.evaluation || res.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Evaluation failed.');
    } finally {
      setEvaluating(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner spinner-lg"></div>
        <p>Loading evaluation interface...</p>
      </div>
    );
  }

  const chartUrl = getChartUrl('model_comparison.png');

  return (
    <div>
      <div className="page-header animate-in">
        <h2>Model Evaluation</h2>
        <p>Step 8 — Calculate BLEU, chrF, and TER metrics, and run baseline comparisons</p>
      </div>

      {/* Select Job / Trigger Eval */}
      <div className="card animate-in" style={{ marginBottom: 24 }}>
        <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--accent-blue-light)' }}>
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="22" y1="12" x2="18" y2="12"></line>
            <line x1="6" y1="12" x2="2" y2="12"></line>
            <line x1="12" y1="6" x2="12" y2="2"></line>
            <line x1="12" y1="22" x2="12" y2="18"></line>
          </svg>
          Select Trained Model Job
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
            <label>Training Job</label>
            <select
              className="form-input"
              value={selectedJobId || ''}
              onChange={(e) => setSelectedJobId(Number(e.target.value))}
            >
              <option value="">-- Select Completed Job --</option>
              {jobs.map((job) => (
                <option key={job.id} value={job.id}>
                  Job #{job.id} (Dataset: {job.dataset_name || `ID ${job.dataset}`}, BLEU: {job.best_val_bleu?.toFixed(2) || 'N/A'})
                </option>
              ))}
            </select>
          </div>
          <button
            className="btn btn-primary"
            onClick={handleRunEvaluation}
            disabled={!selectedJobId || evaluating}
          >
            {evaluating ? (
              <>
                <span className="spinner" style={{ width: 16, height: 16 }}></span>
                Evaluating...
              </>
            ) : (
              'Run Metric Evaluation'
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="card animate-in" style={{ borderColor: 'var(--accent-red)', marginBottom: 24 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--accent-red)' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="8" x2="12" y2="12"></line>
              <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            <span>{error}</span>
          </div>
        </div>
      )}

      {evaluation ? (
        <div className="animate-in">
          {/* Target Status (Exit Condition) */}
          <div
            className="card"
            style={{
              borderColor: evaluation.meets_target ? 'var(--accent-green)' : 'var(--accent-amber)',
              background: evaluation.meets_target
                ? 'rgba(16, 185, 129, 0.04)'
                : 'rgba(245, 158, 11, 0.04)',
              marginBottom: 24,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
              <div style={{ fontSize: 32 }}>
                {evaluation.meets_target ? (
                  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="var(--accent-green)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                    <polyline points="22 4 12 14.01 9 11.01"></polyline>
                  </svg>
                ) : (
                  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="var(--accent-amber)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                    <line x1="12" y1="9" x2="12" y2="13"></line>
                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                  </svg>
                )}
              </div>
              <div>
                <h3 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)' }}>
                  {evaluation.meets_target
                    ? '🎯 Exit Condition Met — Ready for Production'
                    : '🔄 Exit Condition Not Fully Met'}
                </h3>
                <pre style={{
                  marginTop: 12,
                  fontFamily: 'inherit',
                  fontSize: 14,
                  whiteSpace: 'pre-wrap',
                  color: 'var(--text-secondary)',
                  lineHeight: '1.6'
                }}>
                  {evaluation.target_notes}
                </pre>
              </div>
            </div>
          </div>

          {/* Metrics Overview Table */}
          <div className="charts-grid" style={{ marginBottom: 24 }}>
            <div className="card">
              <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--accent-blue-light)' }}>
                  <line x1="18" y1="20" x2="18" y2="10"></line>
                  <line x1="12" y1="20" x2="12" y2="4"></line>
                  <line x1="6" y1="20" x2="6" y2="14"></line>
                </svg>
                Metric Comparison
              </div>
              <table className="data-table" style={{ marginTop: 12 }}>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Pretrained Baseline</th>
                    <th>Our Fine-Tuned Model</th>
                    <th>Difference</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>BLEU Score (sacrebleu) ↑</strong></td>
                    <td>{evaluation.baseline_bleu?.toFixed(2)}</td>
                    <td style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>{evaluation.bleu_score?.toFixed(2)}</td>
                    <td style={{ color: evaluation.bleu_score - evaluation.baseline_bleu >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                      {(evaluation.bleu_score - evaluation.baseline_bleu) >= 0 ? '+' : ''}{(evaluation.bleu_score - evaluation.baseline_bleu).toFixed(2)}
                    </td>
                  </tr>
                  <tr>
                    <td><strong>chrF Score ↑</strong></td>
                    <td>{evaluation.baseline_chrf?.toFixed(2)}</td>
                    <td style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>{evaluation.chrf_score?.toFixed(2)}</td>
                    <td style={{ color: evaluation.chrf_score - evaluation.baseline_chrf >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                      {(evaluation.chrf_score - evaluation.baseline_chrf) >= 0 ? '+' : ''}{(evaluation.chrf_score - evaluation.baseline_chrf).toFixed(2)}
                    </td>
                  </tr>
                  <tr>
                    <td><strong>TER (Translation Edit Rate) ↓</strong></td>
                    <td>{evaluation.baseline_ter?.toFixed(2)}</td>
                    <td style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>{evaluation.ter_score?.toFixed(2)}</td>
                    <td style={{ color: evaluation.baseline_ter - evaluation.ter_score >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                      {/* Note: Lower TER is better, so baseline_ter - ter_score > 0 is good */}
                      {(evaluation.baseline_ter - evaluation.ter_score) >= 0 ? '-' : '+'}{(Math.abs(evaluation.baseline_ter - evaluation.ter_score)).toFixed(2)}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Generated Chart from Django */}
            <div className="chart-container">
              <div className="chart-title">📊 Baseline vs Fine-tuned performance</div>
              <img
                src={`${chartUrl}?t=${new Date().getTime()}`}
                alt="Model comparison chart"
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
            </div>
          </div>

          {/* Qualitative Examples Table */}
          {evaluation.example_translations && evaluation.example_translations.length > 0 && (
            <div className="card" style={{ marginBottom: 24 }}>
              <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--accent-blue-light)' }}>
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
                Qualitative Translation Examples
              </div>
              <div style={{ overflowX: 'auto', marginTop: 12 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th style={{ minWidth: '200px' }}>Source (English)</th>
                      <th style={{ minWidth: '200px' }}>Baseline Model (Pretrained)</th>
                      <th style={{ minWidth: '200px' }}>Fine-Tuned Model</th>
                      <th style={{ minWidth: '200px' }}>Reference (Human Arabic)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evaluation.example_translations.map((ex, idx) => (
                      <tr key={idx}>
                        <td style={{ fontSize: 13, color: 'var(--text-primary)' }}>{ex.source}</td>
                        <td className="arabic-text" style={{ fontSize: 14, color: 'var(--accent-amber)', direction: 'rtl' }}>
                          {ex.baseline_output}
                        </td>
                        <td className="arabic-text" style={{ fontSize: 14, color: 'var(--accent-emerald)', direction: 'rtl', fontWeight: 500 }}>
                          {ex.model_output}
                        </td>
                        <td className="arabic-text" style={{ fontSize: 14, color: 'var(--text-secondary)', direction: 'rtl' }}>
                          {ex.reference}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="card" style={{ textAlign: 'center', padding: 48 }}>
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--text-muted)', marginBottom: 16 }}>
            <line x1="18" y1="20" x2="18" y2="10"></line>
            <line x1="12" y1="20" x2="12" y2="4"></line>
            <line x1="6" y1="20" x2="6" y2="14"></line>
          </svg>
          <p style={{ color: 'var(--text-secondary)' }}>
            No evaluation results for this job yet. Click "Run Metric Evaluation" above.
          </p>
        </div>
      )}
    </div>
  );
}

export default Evaluation;
