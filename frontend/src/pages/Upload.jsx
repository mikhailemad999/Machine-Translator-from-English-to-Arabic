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
          <div className="card-title">✅ Step 1 — Exploration Report</div>

          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Dataset ID</div>
              <div className="metric-value info">{result.dataset_id}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Total Pairs</div>
              <div className="metric-value">{result.total_pairs?.toLocaleString()}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Stored in MongoDB</div>
              <div className="metric-value success">{result.stored_in_mongodb?.toLocaleString()}</div>
            </div>
          </div>

          {result.exploration_report && (
            <>
              <h4 style={{ margin: '20px 0 12px', color: 'var(--text-primary)' }}>📊 Dataset Shape</h4>
              <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
                {result.exploration_report.shape?.rows} rows × {result.exploration_report.shape?.cols} columns
              </p>

              <h4 style={{ margin: '20px 0 12px', color: 'var(--text-primary)' }}>📝 Sample Pairs</h4>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>English</th>
                    <th>Arabic</th>
                  </tr>
                </thead>
                <tbody>
                  {result.exploration_report.sample_pairs?.slice(0, 5).map((pair, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td>{pair.en?.substring(0, 80)}{pair.en?.length > 80 ? '...' : ''}</td>
                      <td className="arabic-text" style={{ direction: 'rtl' }}>
                        {pair.ar?.substring(0, 80)}{pair.ar?.length > 80 ? '...' : ''}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <h4 style={{ margin: '20px 0 12px', color: 'var(--text-primary)' }}>🔍 Encoding Notes</h4>
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
