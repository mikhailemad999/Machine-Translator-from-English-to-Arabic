import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { listDatasets, startTraining, getTrainingStatus, listTrainingJobs } from '../services/api';

/**
 * Training component allowing setting of hyperparameters, starting model
 * fine-tuning runs, and plotting live epoch loss and BLEU graphs.
 */
function Training() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDs, setSelectedDs] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [activeJob, setActiveJob] = useState(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState('');

  // Hyperparameters
  const [batchSize, setBatchSize] = useState(4);
  const [gradAccum, setGradAccum] = useState(8);
  const [lr, setLr] = useState(0.00005);
  const [maxEpochs, setMaxEpochs] = useState(10);
  const [patience, setPatience] = useState(3);

  useEffect(() => {
    listDatasets()
      .then(res => {
        const list = res.data?.results || res.data || [];
        const filtered = list.filter(d => ['uploaded', 'ready', 'completed', 'preprocessed', 'training'].includes(d.status));
        setDatasets(filtered);
        if (filtered.length > 0) setSelectedDs(filtered[0]?.id);
      })
      .catch(() => {});

    listTrainingJobs()
      .then(res => {
        const jobList = res.data?.results || res.data || [];
        setJobs(jobList);
        const running = jobList.find(j => j.status === 'running' || j.status === 'queued');
        if (running) setActiveJob(running);
      })
      .catch(() => {});
  }, []);

  // Poll active job
  useEffect(() => {
    if (!activeJob || !['running', 'queued'].includes(activeJob.status)) return;

    const interval = setInterval(async () => {
      try {
        const res = await getTrainingStatus(activeJob.id);
        setActiveJob(res.data);
        if (res.data.status === 'completed' || res.data.status === 'failed') {
          clearInterval(interval);
          listTrainingJobs().then(r => setJobs(r.data?.results || r.data || [])).catch(() => {});
        }
      } catch (err) {
        clearInterval(interval);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [activeJob?.id, activeJob?.status]);

  const handleStartTraining = async () => {
    if (!selectedDs) return;
    setStarting(true);
    setError('');

    try {
      const res = await startTraining({
        dataset_id: selectedDs,
        batch_size: batchSize,
        gradient_accumulation: gradAccum,
        learning_rate: lr,
        max_epochs: maxEpochs,
        early_stopping_patience: patience,
        fp16: true,
        weight_decay: 0.01,
      });
      setActiveJob({ id: res.data.job_id, status: 'queued', epoch_data: [] });
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to start training.');
    } finally {
      setStarting(false);
    }
  };

  const epochData = activeJob?.epoch_data || [];

  return (
    <div>
      <div className="page-header">
        <h2>Model Training</h2>
        <p>Step 7 — Fine-tune Helsinki-NLP/opus-mt-en-ar with learning curves</p>
      </div>

      {/* Training Config */}
      <div className="card animate-in" style={{ marginBottom: 24 }}>
        <div className="card-title">⚙️ Training Configuration</div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
          <div className="form-group">
            <label>Dataset</label>
            <select className="form-input" value={selectedDs || ''} onChange={e => setSelectedDs(Number(e.target.value))}>
              {datasets.map(ds => (
                <option key={ds.id} value={ds.id}>{ds.name}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Batch Size</label>
            <input className="form-input" type="number" value={batchSize} onChange={e => setBatchSize(Number(e.target.value))} />
          </div>
          <div className="form-group">
            <label>Gradient Accumulation</label>
            <input className="form-input" type="number" value={gradAccum} onChange={e => setGradAccum(Number(e.target.value))} />
          </div>
          <div className="form-group">
            <label>Learning Rate</label>
            <input className="form-input" type="number" step="0.00001" value={lr} onChange={e => setLr(Number(e.target.value))} />
          </div>
          <div className="form-group">
            <label>Max Epochs</label>
            <input className="form-input" type="number" value={maxEpochs} onChange={e => setMaxEpochs(Number(e.target.value))} />
          </div>
          <div className="form-group">
            <label>Early Stopping Patience</label>
            <input className="form-input" type="number" value={patience} onChange={e => setPatience(Number(e.target.value))} />
          </div>
        </div>

        <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
          <button className="btn btn-primary" onClick={handleStartTraining} disabled={starting || !selectedDs}>
            {starting ? (
              <><span className="spinner" style={{ width: 16, height: 16 }}></span> Starting...</>
            ) : '🚀 Start Training'}
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-muted)', fontSize: 13 }}>
            <span>Effective batch: {batchSize * gradAccum}</span>
            <span>•</span>
            <span>fp16: ON</span>
          </div>
        </div>
      </div>

      {error && (
        <div className="card" style={{ borderColor: 'var(--accent-red)', marginBottom: 24 }}>
          <p style={{ color: 'var(--accent-red)' }}>❌ {error}</p>
        </div>
      )}

      {/* Active Job Status */}
      {activeJob && (
        <div className="card animate-in" style={{ marginBottom: 24 }}>
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            🧠 Training Job #{activeJob.id}
            <span className={`badge badge-${
              activeJob.status === 'completed' ? 'success' :
              activeJob.status === 'failed' ? 'error' : 'running'
            }`}>
              {activeJob.status === 'running' && <span className="spinner" style={{ width: 12, height: 12 }}></span>}
              {activeJob.status}
            </span>
          </div>

          {/* Epoch Metrics */}
          {epochData.length > 0 && (
            <>
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">Current Epoch</div>
                  <div className="metric-value info">{epochData[epochData.length - 1]?.epoch}</div>
                  <div className="metric-sub">of {activeJob.max_epochs}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Best Val BLEU</div>
                  <div className={`metric-value ${activeJob.best_val_bleu >= 25 ? 'success' : 'warning'}`}>
                    {activeJob.best_val_bleu?.toFixed(2) || epochData[epochData.length - 1]?.val_bleu?.toFixed(2)}
                  </div>
                  <div className="metric-sub">Target: ≥ 25.0</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Train Loss</div>
                  <div className="metric-value">{epochData[epochData.length - 1]?.train_loss}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Val Loss</div>
                  <div className="metric-value">{epochData[epochData.length - 1]?.val_loss}</div>
                </div>
              </div>

              {/* Learning Curves */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginTop: 24 }}>
                <div>
                  <h4 style={{ marginBottom: 12, fontSize: 14, color: 'var(--text-secondary)' }}>Loss Curves</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={epochData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="epoch" stroke="#6b7280" />
                      <YAxis stroke="#6b7280" />
                      <Tooltip contentStyle={{ background: '#1a1f35', border: '1px solid rgba(255,255,255,0.1)' }} />
                      <Legend />
                      <Line type="monotone" dataKey="train_loss" stroke="#3b82f6" strokeWidth={2} name="Train Loss" dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="val_loss" stroke="#ef4444" strokeWidth={2} name="Val Loss" dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div>
                  <h4 style={{ marginBottom: 12, fontSize: 14, color: 'var(--text-secondary)' }}>BLEU Score</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={epochData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="epoch" stroke="#6b7280" />
                      <YAxis stroke="#6b7280" />
                      <Tooltip contentStyle={{ background: '#1a1f35', border: '1px solid rgba(255,255,255,0.1)' }} />
                      <Legend />
                      <Line type="monotone" dataKey="val_bleu" stroke="#10b981" strokeWidth={2} name="Val BLEU" dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}

          {/* Diagnosis */}
          {activeJob.diagnosis && activeJob.diagnosis !== 'unknown' && (
            <div style={{ marginTop: 20, padding: 16, background: 'var(--bg-input)', borderRadius: 'var(--radius-sm)' }}>
              <div style={{ fontWeight: 600, marginBottom: 8, color: 'var(--text-primary)' }}>
                {activeJob.diagnosis === 'well_fit' ? '✅' : activeJob.diagnosis === 'overfitting' ? '⚠️' : '❌'}{' '}
                Diagnosis: {activeJob.diagnosis.replace('_', ' ')}
              </div>
              <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>{activeJob.diagnosis_notes}</p>
            </div>
          )}
        </div>
      )}

      {/* Previous Jobs */}
      {jobs.length > 0 && (
        <div className="card animate-in">
          <div className="card-title">📋 Training History</div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Job ID</th>
                <th>Status</th>
                <th>Best BLEU</th>
                <th>Epochs</th>
                <th>Diagnosis</th>
                <th>Started</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map(job => (
                <tr key={job.id} onClick={() => setActiveJob(job)} style={{ cursor: 'pointer' }}>
                  <td>#{job.id}</td>
                  <td>
                    <span className={`badge badge-${job.status === 'completed' ? 'success' : job.status === 'failed' ? 'error' : 'running'}`}>
                      {job.status}
                    </span>
                  </td>
                  <td>{job.best_val_bleu?.toFixed(2) || '—'}</td>
                  <td>{job.epoch_data?.length || 0}/{job.max_epochs}</td>
                  <td>{job.diagnosis || '—'}</td>
                  <td style={{ color: 'var(--text-muted)' }}>
                    {new Date(job.started_at).toLocaleString()}
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

export default Training;
