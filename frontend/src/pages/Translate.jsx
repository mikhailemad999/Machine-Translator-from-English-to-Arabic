import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { translateText, translateBatch, getTranslationHistory, listTrainingJobs } from '../services/api';

/**
 * Parses raw CSV string data into a 2D array of rows and columns,
 * handling double quotes and escaping according to RFC-4180.
 *
 * @param {string} text Raw CSV text content.
 * @returns {Array<Array<string>>} List of parsed rows.
 */
function parseCSVText(text) {
  const lines = [];
  let row = [""];
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    const next = text[i+1];

    if (c === '"') {
      if (inQuotes && next === '"') {
        row[row.length - 1] += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (c === ',' && !inQuotes) {
      row.push('');
    } else if ((c === '\r' || c === '\n') && !inQuotes) {
      if (c === '\r' && next === '\n') {
        i++;
      }
      lines.push(row);
      row = [''];
    } else {
      row[row.length - 1] += c;
    }
  }
  if (row.length > 1 || row[0] !== '') {
    lines.push(row);
  }
  return lines;
}

/**
 * Translate component serving the interactive text translation playground.
 * Supports typing single sentences or uploading files for batch translation.
 */
function Translate() {
  // Tab control
  const [activeTab, setActiveTab] = useState('single'); // 'single' or 'file'

  // Common Model states
  const [jobs, setJobs] = useState([]);
  const [selectedModelPath, setSelectedModelPath] = useState('');
  const [error, setError] = useState('');

  // Single Text translation states
  const [sourceText, setSourceText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);

  // File translation states
  const [csvFile, setCsvFile] = useState(null);
  const [csvHeaders, setCsvHeaders] = useState([]);
  const [csvRows, setCsvRows] = useState([]);
  const [srcColIndex, setSrcColIndex] = useState(0);
  const [translatingFile, setTranslatingFile] = useState(false);
  const [translatedCsvRows, setTranslatedCsvRows] = useState([]);
  const [translateProgress, setTranslateProgress] = useState(0);
  const [translateCount, setTranslateCount] = useState(0);
  const [previewRows, setPreviewRows] = useState([]);

  useEffect(() => {
    loadHistory();

    listTrainingJobs()
      .then((res) => {
        const completed = (res.data?.results || res.data || []).filter(
          (j) => j.status === 'completed'
        );
        setJobs(completed);
        if (completed.length > 0) {
          setSelectedModelPath(completed[0].model_checkpoint_path);
        }
      })
      .catch((err) => console.error('Error fetching jobs:', err));
  }, []);

  const loadHistory = () => {
    getTranslationHistory(10)
      .then((res) => setHistory(res.data?.results || res.data || []))
      .catch((err) => console.error('Error fetching history:', err));
  };

  // Single sentence translation
  const handleTranslate = async () => {
    if (!sourceText.trim()) return;
    setLoading(true);
    setError('');
    setStats(null);

    try {
      const res = await translateText(sourceText, selectedModelPath);
      setTranslatedText(res.data.translated_text);
      setStats({
        timeMs: res.data.translation_time_ms,
        model: res.data.model_used,
      });
      loadHistory();
    } catch (err) {
      setError(err.response?.data?.error || 'Translation request failed.');
    } finally {
      setLoading(false);
    }
  };

  // File batch processing
  const handleFileSelect = (file) => {
    setCsvFile(file);
    setTranslatedCsvRows([]);
    setPreviewRows([]);
    setTranslateProgress(0);
    setTranslateCount(0);
    setError('');

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const parsed = parseCSVText(text);
      if (parsed.length > 0) {
        const headers = parsed[0];
        const rows = parsed.slice(1).filter(r => r.some(cell => cell.trim() !== '')); // skip empty rows
        setCsvHeaders(headers);
        setCsvRows(rows);

        // Pre-select English column candidate
        const enIdx = headers.findIndex(h => 
          /^(en|english|text|source|src|sentence|eng)$/i.test(h.trim())
        );
        setSrcColIndex(enIdx >= 0 ? enIdx : 0);
      }
    };
    reader.readAsText(file, 'utf-8');
  };

  const handleTranslateFile = async () => {
    if (!csvFile || csvRows.length === 0) return;
    setTranslatingFile(true);
    setError('');
    setTranslateProgress(0);
    setTranslateCount(0);
    setTranslatedCsvRows([]);
    setPreviewRows([]);

    const batchSize = 16;
    const totalRows = csvRows.length;
    const translated = [];

    try {
      for (let i = 0; i < totalRows; i += batchSize) {
        const batch = csvRows.slice(i, i + batchSize);
        const texts = batch.map(row => row[srcColIndex] || '');
        
        const res = await translateBatch(texts, selectedModelPath);
        const translations = res.data.translations || [];

        batch.forEach((row, idx) => {
          const transVal = translations[idx] || '';
          translated.push([...row, transVal]);
        });

        const currentCount = translated.length;
        setTranslateCount(currentCount);
        setTranslateProgress(Math.round((currentCount / totalRows) * 100));
        setTranslatedCsvRows([...translated]);
      }
      setPreviewRows(translated.slice(0, 10));
    } catch (err) {
      setError(err.response?.data?.error || 'Batch translation failed. Please check backend connection.');
    } finally {
      setTranslatingFile(false);
    }
  };

  const handleDownload = () => {
    // Generate CSV content with Excel friendly UTF-8 BOM
    const headersLine = [...csvHeaders, 'arabic_translation'].map(h => `"${h.replace(/"/g, '""')}"`).join(',');
    const rowsLines = translatedCsvRows.map(row => 
      row.map(cell => `"${(cell ?? '').toString().replace(/"/g, '""')}"`).join(',')
    );
    const csvContent = [headersLine, ...rowsLines].join('\n');
    
    const blob = new Blob([new Uint8Array([0xEF, 0xBB, 0xBF]), csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', csvFile.name.replace(/\.[^/.]+$/, '_translated.csv'));
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'text/csv': ['.csv'],
      'text/tab-separated-values': ['.tsv'],
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        handleFileSelect(acceptedFiles[0]);
      }
    },
  });

  return (
    <div>
      <div className="page-header animate-in">
        <h2>Interactive Translator</h2>
        <p>Translate English sentences or process bulk CSV datasets using your machine translator</p>
      </div>

      {/* Model Selection */}
      <div className="card animate-in" style={{ marginBottom: 24 }}>
        <div className="form-group" style={{ marginBottom: 0 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--accent-blue-light)' }}>
              <rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect>
              <rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect>
              <line x1="6" y1="6" x2="6.01" y2="6"></line>
              <line x1="6" y1="18" x2="6.01" y2="18"></line>
            </svg>
            Active Model Checkpoint
          </label>
          <select
            className="form-input"
            value={selectedModelPath}
            onChange={(e) => setSelectedModelPath(e.target.value)}
          >
            <option value="">Baseline Pretrained Model (Helsinki-NLP/opus-mt-en-ar)</option>
            {jobs.map((job) => (
              <option key={job.id} value={job.model_checkpoint_path}>
                Fine-tuned Checkpoint #{job.id} (BLEU: {job.best_val_bleu?.toFixed(2) || 'N/A'})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Tab Selector */}
      <div className="tab-menu animate-in" style={{ display: 'flex', gap: 12, marginBottom: 24, borderBottom: '1px solid var(--border-color)', paddingBottom: 16 }}>
        <button
          className={`btn ${activeTab === 'single' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => { setActiveTab('single'); setError(''); }}
        >
          ✍️ Translate Single Text
        </button>
        <button
          className={`btn ${activeTab === 'file' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => { setActiveTab('file'); setError(''); }}
        >
          📁 Translate CSV File
        </button>
      </div>

      {error && (
        <div className="card animate-in" style={{ borderColor: 'var(--accent-red)', marginBottom: 24 }}>
          <p style={{ color: 'var(--accent-red)' }}>❌ {error}</p>
        </div>
      )}

      {/* Single translation tab */}
      {activeTab === 'single' && (
        <div className="translation-container animate-in">
          {/* English Input */}
          <div className="translation-box">
            <div className="lang-label">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="2" y1="12" x2="22" y2="12"></line>
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
              </svg>
              English Source
            </div>
            <textarea
              value={sourceText}
              onChange={(e) => setSourceText(e.target.value)}
              placeholder="Type English text to translate..."
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleTranslate();
                }
              }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 12, alignItems: 'center' }}>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                Press Enter to translate, Shift+Enter for new line.
              </span>
              <button
                className="btn btn-primary"
                onClick={handleTranslate}
                disabled={loading || !sourceText.trim()}
              >
                {loading ? (
                  <>
                    <span className="spinner" style={{ width: 14, height: 14 }}></span> Translating...
                  </>
                ) : (
                  'Translate'
                )}
              </button>
            </div>
          </div>

          {/* Arabic Output */}
          <div className="translation-box">
            <div className="lang-label" style={{ color: 'var(--accent-emerald)' }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="2" y1="12" x2="22" y2="12"></line>
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
              </svg>
              Arabic Translation
            </div>
            <div className="output-text">
              {translatedText || <span style={{ color: 'var(--text-muted)', fontSize: 16 }}>Arabic translation will appear here...</span>}
            </div>
            {stats && (
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 12, fontSize: 12, color: 'var(--text-muted)' }}>
                <span>Latency: <strong style={{ color: 'var(--text-primary)' }}>{stats.timeMs} ms</strong></span>
                <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', maxWidth: '300px' }}>
                  Model: <strong style={{ color: 'var(--text-primary)' }}>{stats.model}</strong>
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* CSV translation tab */}
      {activeTab === 'file' && (
        <div className="animate-in">
          <div className="card" style={{ marginBottom: 24 }}>
            <div className="card-title">📁 Upload English CSV/TSV File</div>
            <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
              <input {...getInputProps()} />
              <div className="dropzone-icon">📥</div>
              {csvFile ? (
                <p style={{ color: 'var(--accent-green)', fontWeight: 600 }}>
                  ✅ Selected: {csvFile.name} ({csvRows.length} rows loaded)
                </p>
              ) : (
                <p>Drag & drop your CSV or TSV file here, or click to browse</p>
              )}
              <div className="dropzone-hint">
                Must contain a column of English text to translate.
              </div>
            </div>

            {csvFile && csvHeaders.length > 0 && (
              <div style={{ marginTop: 24 }}>
                <div className="form-group" style={{ maxWidth: 400 }}>
                  <label>Select English Column</label>
                  <select
                    className="form-input"
                    value={srcColIndex}
                    onChange={(e) => setSrcColIndex(Number(e.target.value))}
                  >
                    {csvHeaders.map((header, idx) => (
                      <option key={idx} value={idx}>
                        {header} (Example: "{csvRows[0]?.[idx]?.substring(0, 40) || ''}")
                      </option>
                    ))}
                  </select>
                </div>

                <button
                  className="btn btn-primary"
                  onClick={handleTranslateFile}
                  disabled={translatingFile || csvRows.length === 0}
                >
                  {translatingFile ? (
                    <>
                      <span className="spinner" style={{ width: 14, height: 14 }}></span> Translating ({translateCount}/{csvRows.length})...
                    </>
                  ) : (
                    '🚀 Translate Full File'
                  )}
                </button>
              </div>
            )}
          </div>

          {/* Real-time Progress Bar */}
          {translatingFile && (
            <div className="card animate-in" style={{ marginBottom: 24 }}>
              <div className="card-title">⚡ Batch Translation Progress</div>
              <p style={{ color: 'var(--text-secondary)', marginBottom: 12 }}>
                Translating sentence {translateCount} of {csvRows.length} ({translateProgress}%)
              </p>
              <div style={{ width: '100%', height: 10, background: 'var(--bg-input)', borderRadius: 5, overflow: 'hidden' }}>
                <div style={{ width: `${translateProgress}%`, height: '100%', background: 'var(--gradient-primary)', transition: 'width 0.2s ease' }}></div>
              </div>
            </div>
          )}

          {/* Translated Preview & Download */}
          {translatedCsvRows.length > 0 && !translatingFile && (
            <div className="card animate-in">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
                <div className="card-title" style={{ marginBottom: 0 }}>🎉 Translation Complete!</div>
                <button className="btn btn-success" onClick={handleDownload}>
                  📥 Download Translated CSV ({translatedCsvRows.length} rows)
                </button>
              </div>

              <h4 style={{ margin: '20px 0 12px', color: 'var(--text-primary)' }}>📝 Translation Preview (First 10 Rows)</h4>
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>English Source ({csvHeaders[srcColIndex]})</th>
                      <th>Arabic Translation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {previewRows.map((row, i) => (
                      <tr key={i}>
                        <td>{i + 1}</td>
                        <td style={{ fontSize: 13 }}>{row[srcColIndex]}</td>
                        <td className="arabic-text" style={{ fontSize: 14, color: 'var(--accent-emerald)', direction: 'rtl', fontWeight: 500 }}>
                          {row[row.length - 1]}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Translation History */}
      {history.length > 0 && activeTab === 'single' && (
        <div className="card animate-in" style={{ marginTop: 24 }}>
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--accent-blue-light)' }}>
              <circle cx="12" cy="12" r="10"></circle>
              <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
            Recent Translations
          </div>
          <table className="data-table" style={{ marginTop: 12 }}>
            <thead>
              <tr>
                <th style={{ width: '40%' }}>English Source</th>
                <th style={{ width: '40%' }}>Arabic Translation</th>
                <th style={{ width: '10%' }}>Latency</th>
                <th style={{ width: '10%' }}>Date</th>
              </tr>
            </thead>
            <tbody>
              {history.map((log) => (
                <tr key={log.id}>
                  <td>{log.source_text}</td>
                  <td className="arabic-text" style={{ direction: 'rtl', fontWeight: 500 }}>
                    {log.translated_text}
                  </td>
                  <td>{log.translation_time_ms} ms</td>
                  <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                    {new Date(log.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
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

export default Translate;

