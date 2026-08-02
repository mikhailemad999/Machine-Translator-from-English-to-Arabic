import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { uploadDataset, downloadSampleDataset } from '../services/api';

/**
 * Upload component handling local file selection (drag & drop),
 * specifying column names, and triggering dataset upload to backend APIs.
 * Also supports downloading and importing sample datasets.
 */
function Upload() {
  const [file, setFile] = useState(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [enColumn, setEnColumn] = useState('en');
  const [arColumn, setArColumn] = useState('ar');
  const [uploading, setUploading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'text/csv': ['.csv'],
      'text/tab-separated-values': ['.tsv'],
      'application/json': ['.json'],
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0]);
        if (!name) setName(acceptedFiles[0].name.replace(/\.[^/.]+$/, ''));
      }
    },
  });

  const handleUpload = async () => {
    if (!file || !name) {
      setError('Please select a file and enter a name.');
      return;
    }

    setUploading(true);
    setError('');
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);
    formData.append('description', description);
    formData.append('en_column', enColumn);
    formData.append('ar_column', arColumn);

    try {
      const res = await uploadDataset(formData);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const handleDownloadSample = async () => {
    setDownloading(true);
    setError('');
    setResult(null);

    try {
      const res = await downloadSampleDataset(10000);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Download failed.');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h2>Upload Dataset</h2>
        <p>Step 1 — Load your EN-AR parallel corpus or download a sample</p>
      </div>

      {/* Quick Download */}
      <div className="card animate-in" style={{ marginBottom: 24 }}>
        <div className="card-title">⚡ Quick Start — Download Sample Dataset</div>
        <p style={{ color: 'var(--text-secondary)', marginBottom: 16, fontSize: 14 }}>
          Don't have a dataset? Download 10,000 EN-AR sentence pairs from OPUS Books.
        </p>
        <button
          className="btn btn-success"
          onClick={handleDownloadSample}
          disabled={downloading}
        >
          {downloading ? (
            <><span className="spinner" style={{ width: 16, height: 16 }}></span> Downloading...</>
          ) : '📥 Download OPUS Books EN-AR'}
        </button>
      </div>

      {/* File Upload */}
      <div className="card animate-in" style={{ marginBottom: 24 }}>
        <div className="card-title">📁 Upload Your Dataset</div>

        <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
          <input {...getInputProps()} />
          <div className="dropzone-icon">📂</div>
          {file ? (
            <p style={{ color: 'var(--accent-green)', fontWeight: 600 }}>
              ✅ {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </p>
          ) : isDragActive ? (
            <p>Drop the file here...</p>
          ) : (
            <p>Drag & drop a CSV, TSV, or JSON file here, or click to browse</p>
          )}
          <div className="dropzone-hint">
            Supported: .csv, .tsv, .json • Max 100MB
          </div>
        </div>

        {/* Column Mapping */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 20 }}>
          <div className="form-group">
            <label>Dataset Name</label>
            <input
              className="form-input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My EN-AR Dataset"
            />
          </div>
          <div className="form-group">
            <label>Description (optional)</label>
            <input
              className="form-input"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Parallel corpus for MT training"
            />
          </div>
          <div className="form-group">
            <label>English Column Name</label>
            <input
              className="form-input"
              value={enColumn}
              onChange={(e) => setEnColumn(e.target.value)}
              placeholder="en"
            />
          </div>
          <div className="form-group">
            <label>Arabic Column Name</label>
            <input
              className="form-input"
              value={arColumn}
              onChange={(e) => setArColumn(e.target.value)}
              placeholder="ar"
            />
          </div>
        </div>

        <button
          className="btn btn-primary"
          onClick={handleUpload}
          disabled={!file || !name || uploading}
          style={{ marginTop: 8 }}
        >
          {uploading ? (
            <><span className="spinner" style={{ width: 16, height: 16 }}></span> Uploading...</>
          ) : '🚀 Upload & Explore'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="card animate-in" style={{ borderColor: 'var(--accent-red)', marginBottom: 24 }}>
          <p style={{ color: 'var(--accent-red)' }}>❌ {error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="card animate-in">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
            <div className="card-title" style={{ marginBottom: 0 }}>✅ Step 1 — Exploration & Database Analysis</div>
            {result.exploration_report?.dataset_health_score !== undefined && (
              <span className={`badge ${result.exploration_report.dataset_health_score >= 80 ? 'badge-success' : 'badge-warning'}`} style={{ fontSize: 14, padding: '6px 12px' }}>
                Health Score: {result.exploration_report.dataset_health_score}%
              </span>
            )}
          </div>

          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Dataset ID</div>
              <div className="metric-value info">{result.dataset_id}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Total Sentence Pairs</div>
              <div className="metric-value">{result.total_pairs?.toLocaleString()}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Stored in Storage</div>
              <div className="metric-value success">{result.stored_in_mongodb?.toLocaleString()}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">RAM Footprint</div>
              <div className="metric-value">{result.exploration_report?.shape?.memory_mb || '0.1'} MB</div>
            </div>
          </div>

          {result.exploration_report && (
            <>
              {/* Database & Data Integrity Metrics */}
              {result.exploration_report.database_metrics && (
                <div style={{ marginTop: 24 }}>
                  <h4 style={{ marginBottom: 12, color: 'var(--text-primary)' }}>🗄️ Database & Integrity Metrics</h4>
                  <div className="metrics-grid">
                    <div className="metric-card">
                      <div className="metric-label">Exact Duplicates</div>
                      <div className="metric-value warning">
                        {result.exploration_report.database_metrics.exact_duplicates?.toLocaleString()} ({result.exploration_report.database_metrics.duplicate_pct}%)
                      </div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">Empty EN Rows</div>
                      <div className="metric-value">{result.exploration_report.database_metrics.empty_rows_en}</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">Empty AR Rows</div>
                      <div className="metric-value">{result.exploration_report.database_metrics.empty_rows_ar}</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">Target Storage Collection</div>
                      <div className="metric-value info" style={{ fontSize: 13, wordBreak: 'break-all' }}>
                        {result.mongo_collection}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Column Statistical Analysis */}
              {result.exploration_report.summary && (
                <div style={{ marginTop: 24 }}>
                  <h4 style={{ marginBottom: 12, color: 'var(--text-primary)' }}>📈 Statistical Length Analysis</h4>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Column</th>
                        <th>Non-Null</th>
                        <th>Unique %</th>
                        <th>Avg / Min / Max Chars</th>
                        <th>Avg / Min / Max Tokens</th>
                        <th>Median / Std Tokens</th>
                      </tr>
                    </thead>
                    <tbody>
                      {['en', 'ar'].map(col => {
                        const s = result.exploration_report.summary[col];
                        if (!s) return null;
                        return (
                          <tr key={col}>
                            <td style={{ fontWeight: 600, textTransform: 'uppercase' }}>{col}</td>
                            <td>{s.non_null?.toLocaleString()}</td>
                            <td>{s.unique_pct}%</td>
                            <td>{s.avg_length_chars} ({s.min_length_chars} - {s.max_length_chars})</td>
                            <td>{s.avg_length_tokens} ({s.min_length_tokens} - {s.max_length_tokens})</td>
                            <td>{s.median_tokens} (±{s.std_tokens})</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Cross-Language Ratios */}
              {result.exploration_report.length_ratios && (
                <div style={{ display: 'flex', gap: 24, marginTop: 16, fontSize: 14, color: 'var(--text-secondary)' }}>
                  <div>🔤 <strong>EN / AR Character Ratio:</strong> {result.exploration_report.length_ratios.char_ratio_en_to_ar}</div>
                  <div>🧩 <strong>EN / AR Token Ratio:</strong> {result.exploration_report.length_ratios.token_ratio_en_to_ar}</div>
                </div>
              )}

              {/* 50 Sample Sentence Pairs Preview */}
              <div style={{ marginTop: 24 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                  <h4 style={{ margin: 0, color: 'var(--text-primary)' }}>📝 Sample Translation Pairs Preview (50 Rows)</h4>
                  <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                    Showing {Math.min(50, result.exploration_report.sample_pairs?.length || 0)} pairs
                  </span>
                </div>

                <div style={{ maxHeight: '420px', overflowY: 'auto', border: '1px solid var(--border-color)', borderRadius: 8 }}>
                  <table className="data-table" style={{ margin: 0 }}>
                    <thead style={{ position: 'sticky', top: 0, background: 'var(--bg-card)', zIndex: 1 }}>
                      <tr>
                        <th style={{ width: 60 }}>#</th>
                        <th style={{ width: '45%' }}>English Sentence</th>
                        <th style={{ width: '45%' }}>Arabic Sentence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.exploration_report.sample_pairs?.slice(0, 50).map((pair, i) => (
                        <tr key={i}>
                          <td style={{ color: 'var(--text-secondary)', fontSize: 12 }}>{i + 1}</td>
                          <td>{pair.en}</td>
                          <td className="arabic-text" style={{ direction: 'rtl' }}>
                            {pair.ar}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <h4 style={{ margin: '24px 0 12px', color: 'var(--text-primary)' }}>🔍 Encoding & Script Check</h4>
              <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
                {result.exploration_report.encoding_notes}
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default Upload;
