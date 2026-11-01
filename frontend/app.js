/**
 * Fake News Detection System - Academic Frontend Controller
 * Supports Direct Text, Document (PDF/DOCX/TXT) Upload, and Website URL Extraction.
 * Interacts with pre-trained Colab artifacts and verified metrics.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Determine API base URL dynamically
  const API_BASE = (window.location.protocol.startsWith('http') && window.location.port === '5000')
    ? ''
    : 'http://127.0.0.1:5000';

  // State Management
  let currentMode = 'text'; // 'text' | 'document' | 'url'
  let currentFile = null;
  let currentDomain = '';
  let chartInstance = null;

  // DOM Elements - Status & Form
  const backendStatusCard = document.getElementById('backend-status-card');
  const backendStatusText = document.getElementById('backend-status-text');

  const newsForm = document.getElementById('news-form');
  const modelSelector = document.getElementById('model-selector');
  const modelDescription = document.getElementById('model-description');
  const btnAnalyze = document.getElementById('btn-analyze');
  const btnClear = document.getElementById('btn-clear');
  const analyzeSpinner = document.getElementById('analyze-spinner');
  const analyzeBtnText = document.getElementById('analyze-btn-text');
  const formError = document.getElementById('form-error');
  const formErrorText = document.getElementById('form-error-text');

  // Tabs
  const tabBtns = document.querySelectorAll('.tab-btn');
  const paneText = document.getElementById('pane-text');
  const paneDocument = document.getElementById('pane-document');
  const paneUrl = document.getElementById('pane-url');

  // Pane 1: Text Elements
  const newsTitleInput = document.getElementById('news-title');
  const newsTextInput = document.getElementById('news-text');
  const charWordCounter = document.getElementById('char-word-counter');
  const btnSampleReal = document.getElementById('btn-sample-real');
  const btnSampleFake = document.getElementById('btn-sample-fake');
  const btnSamplePolitical = document.getElementById('btn-sample-political');

  // Pane 2: Document Elements
  const docDropzone = document.getElementById('doc-dropzone');
  const docFileInput = document.getElementById('doc-file-input');
  const docPreviewBox = document.getElementById('doc-preview-box');
  const docFilename = document.getElementById('doc-filename');
  const docBadgeType = document.getElementById('doc-badge-type');
  const docFilesize = document.getElementById('doc-filesize');
  const docStats = document.getElementById('doc-stats');
  const btnRemoveDoc = document.getElementById('btn-remove-doc');
  const docExtractedTitle = document.getElementById('doc-extracted-title');
  const docExtractedText = document.getElementById('doc-extracted-text');
  const docCharWordCounter = document.getElementById('doc-char-word-counter');

  // Pane 3: URL Elements
  const newsUrlInput = document.getElementById('news-url');
  const btnFetchUrl = document.getElementById('btn-fetch-url');
  const fetchSpinner = document.getElementById('fetch-spinner');
  const fetchBtnText = document.getElementById('fetch-btn-text');
  const urlPreviewBox = document.getElementById('url-preview-box');
  const urlExtractedDomain = document.getElementById('url-extracted-domain');
  const urlStats = document.getElementById('url-stats');
  const urlExtractedTitle = document.getElementById('url-extracted-title');
  const urlExtractedText = document.getElementById('url-extracted-text');
  const urlCharWordCounter = document.getElementById('url-char-word-counter');
  const btnUrlToi = document.getElementById('btn-url-toi');
  const btnUrlReuters = document.getElementById('btn-url-reuters');
  const btnUrlAp = document.getElementById('btn-url-ap');

  // Result Elements
  const resultSection = document.getElementById('result-section');
  const resultCard = document.getElementById('result-card');
  const resultVerdict = document.getElementById('result-verdict');
  const resultModelName = document.getElementById('result-model-name');
  const resultModelBadge = document.getElementById('result-model-badge');
  const resultSourceBadge = document.getElementById('result-source-badge');
  const resultTokens = document.getElementById('result-tokens');
  const resultRawClass = document.getElementById('result-raw-class');
  const resultExplanationText = document.getElementById('result-explanation-text');

  // Performance Section Elements
  const metricsTableBody = document.getElementById('metrics-table-body');
  const metricsError = document.getElementById('metrics-error');
  const metricsContainer = document.getElementById('metrics-container');

  // Model Descriptions Map
  const MODEL_DESCRIPTIONS = {
    logistic_regression: 'Linear classification model for text classification.',
    svm: 'Linear Support Vector Machine for high-dimensional text features.',
    random_forest: 'Ensemble classification model using multiple decision trees.'
  };

  // ==========================================================================
  // 1. Tab Switching Controller
  // ==========================================================================
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode;
      switchTabMode(mode);
    });
  });

  function switchTabMode(mode) {
    currentMode = mode;
    hideError();

    tabBtns.forEach(btn => {
      const isActive = btn.dataset.mode === mode;
      btn.classList.toggle('active', isActive);
      btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });

    paneText.style.display = (mode === 'text') ? 'flex' : 'none';
    paneDocument.style.display = (mode === 'document') ? 'flex' : 'none';
    paneUrl.style.display = (mode === 'url') ? 'flex' : 'none';
  }

  // ==========================================================================
  // 2. Counters Helper
  // ==========================================================================
  function updateTextCounters(titleVal, textVal, counterElem) {
    const combined = (titleVal.trim() + ' ' + textVal.trim()).trim();
    const charCount = combined.length;
    const wordCount = combined ? combined.split(/\s+/).length : 0;
    if (counterElem) {
      counterElem.textContent = `${wordCount} words | ${charCount} characters`;
    }
    return { wordCount, charCount };
  }

  newsTitleInput.addEventListener('input', () => {
    updateTextCounters(newsTitleInput.value, newsTextInput.value, charWordCounter);
    hideError();
  });

  newsTextInput.addEventListener('input', () => {
    updateTextCounters(newsTitleInput.value, newsTextInput.value, charWordCounter);
    hideError();
  });

  docExtractedTitle.addEventListener('input', () => {
    updateTextCounters(docExtractedTitle.value, docExtractedText.value, docCharWordCounter);
    hideError();
  });

  docExtractedText.addEventListener('input', () => {
    updateTextCounters(docExtractedTitle.value, docExtractedText.value, docCharWordCounter);
    hideError();
  });

  urlExtractedTitle.addEventListener('input', () => {
    updateTextCounters(urlExtractedTitle.value, urlExtractedText.value, urlCharWordCounter);
    hideError();
  });

  urlExtractedText.addEventListener('input', () => {
    updateTextCounters(urlExtractedTitle.value, urlExtractedText.value, urlCharWordCounter);
    hideError();
  });

  // Model Selector Event
  modelSelector.addEventListener('change', () => {
    const selected = modelSelector.value;
    modelDescription.textContent = MODEL_DESCRIPTIONS[selected] || '';
  });

  // ==========================================================================
  // 3. Document Drag & Drop / File Extraction
  // ==========================================================================
  ['dragenter', 'dragover'].forEach(eventName => {
    docDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      docDropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    docDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      docDropzone.classList.remove('dragover');
    }, false);
  });

  docDropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleDocumentFile(files[0]);
    }
  });

  docFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleDocumentFile(e.target.files[0]);
    }
  });

  async function handleDocumentFile(file) {
    hideError();

    const allowedExtensions = ['pdf', 'docx', 'txt'];
    const ext = file.name.split('.').pop().toLowerCase();

    if (!allowedExtensions.includes(ext)) {
      showError(`Unsupported file format (.${ext}). Please upload a .pdf, .docx, or .txt file.`);
      return;
    }

    if (file.size > 16 * 1024 * 1024) {
      showError('File size exceeds the 16 MB limit. Please select a smaller file.');
      return;
    }

    currentFile = file;

    // Show temporary upload status
    const dropzoneTitle = docDropzone.querySelector('.dropzone-title');
    const originalTitle = dropzoneTitle.textContent;
    dropzoneTitle.textContent = `Extracting text from ${file.name}...`;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/api/extract-file`, {
        method: 'POST',
        body: formData
      });

      const resData = await response.json();

      if (!response.ok) {
        throw new Error(resData.error || 'Failed to extract text from document.');
      }

      const extracted = resData.data;

      // Populate file info card
      docFilename.textContent = extracted.filename;
      docBadgeType.textContent = extracted.file_type;
      docFilesize.textContent = formatBytes(file.size);

      // Icon by type
      const iconMap = { 'PDF': '📕', 'DOCX': '📘', 'TXT': '📄' };
      document.getElementById('doc-file-icon').textContent = iconMap[extracted.file_type] || '📄';

      // Populate extracted fields
      docExtractedTitle.value = extracted.title || file.name.replace(/\.[^/.]+$/, '');
      docExtractedText.value = extracted.text || '';

      const { wordCount } = updateTextCounters(docExtractedTitle.value, docExtractedText.value, docCharWordCounter);
      docStats.textContent = `${wordCount} words extracted`;

      // Show preview, hide dropzone
      docDropzone.style.display = 'none';
      docPreviewBox.style.display = 'flex';
      resultSection.style.display = 'none';

    } catch (err) {
      showError(err.message || 'Error extracting document content.');
      docFileInput.value = '';
      currentFile = null;
    } finally {
      dropzoneTitle.textContent = originalTitle;
    }
  }

  btnRemoveDoc.addEventListener('click', () => {
    currentFile = null;
    docFileInput.value = '';
    docExtractedTitle.value = '';
    docExtractedText.value = '';
    docPreviewBox.style.display = 'none';
    docDropzone.style.display = 'block';
    hideError();
    resultSection.style.display = 'none';
  });

  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  // ==========================================================================
  // 4. Website Link (URL) Extraction
  // ==========================================================================
  btnFetchUrl.addEventListener('click', () => {
    fetchArticleFromUrl();
  });

  newsUrlInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      fetchArticleFromUrl();
    }
  });

  async function fetchArticleFromUrl(customUrl = null) {
    hideError();
    const url = (customUrl || newsUrlInput.value).trim();

    if (!url) {
      showError('Please enter a valid news website link (e.g. https://www.reuters.com/...).');
      newsUrlInput.focus();
      return;
    }

    btnFetchUrl.disabled = true;
    fetchSpinner.style.display = 'inline-block';
    fetchBtnText.textContent = 'Fetching...';

    try {
      const response = await fetch(`${API_BASE}/api/extract-url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url })
      });

      const resData = await response.json();

      if (!response.ok) {
        throw new Error(resData.error || 'Failed to fetch article from website link.');
      }

      const extracted = resData.data;
      currentDomain = extracted.domain;

      urlExtractedDomain.textContent = extracted.domain;
      urlExtractedTitle.value = extracted.title;
      urlExtractedText.value = extracted.text;

      const { wordCount } = updateTextCounters(urlExtractedTitle.value, urlExtractedText.value, urlCharWordCounter);
      urlStats.textContent = `${wordCount} words extracted`;

      urlPreviewBox.style.display = 'flex';
      resultSection.style.display = 'none';

    } catch (err) {
      showError(err.message || 'Error fetching article from URL.');
      urlPreviewBox.style.display = 'none';
    } finally {
      btnFetchUrl.disabled = false;
      fetchSpinner.style.display = 'none';
      fetchBtnText.textContent = 'Fetch Article';
    }
  }

  // Demo URL shortcuts
  if (btnUrlToi) {
    btnUrlToi.addEventListener('click', () => {
      const toiUrl = 'https://timesofindia.indiatimes.com/defence/news/air-marshal-inderpal-singh-walia-appointed-iaf-vice-chief/articleshow/135363738.cms';
      newsUrlInput.value = toiUrl;
      fetchArticleFromUrl(toiUrl);
    });
  }

  if (btnUrlReuters) {
    btnUrlReuters.addEventListener('click', () => {
      newsUrlInput.value = 'https://en.wikipedia.org/wiki/Reuters';
      fetchArticleFromUrl('https://en.wikipedia.org/wiki/Reuters');
    });
  }

  if (btnUrlAp) {
    btnUrlAp.addEventListener('click', () => {
      newsUrlInput.value = 'https://en.wikipedia.org/wiki/Associated_Press';
      fetchArticleFromUrl('https://en.wikipedia.org/wiki/Associated_Press');
    });
  }

  // ==========================================================================
  // 5. Backend Health Check
  // ==========================================================================
  async function checkHealth() {
    try {
      const response = await fetch(`${API_BASE}/api/health`, { method: 'GET' });
      if (response.ok) {
        backendStatusCard.className = 'status-indicator status-online';
        backendStatusText.textContent = 'Connected';
        return true;
      } else {
        throw new Error('Non-200 status');
      }
    } catch (err) {
      backendStatusCard.className = 'status-indicator status-offline';
      backendStatusText.textContent = 'Offline';
      return false;
    }
  }

  checkHealth();
  setInterval(checkHealth, 8000);

  // ==========================================================================
  // 6. Evaluation Metrics & Performance Chart
  // ==========================================================================
  async function loadMetrics() {
    try {
      const response = await fetch(`${API_BASE}/api/metrics`, { method: 'GET' });
      if (!response.ok) {
        throw new Error('Metrics endpoint error');
      }
      const json = await response.json();
      if (!json || !Array.isArray(json.data) || json.data.length === 0) {
        throw new Error('Empty metrics data');
      }

      renderMetricsTable(json.data);
      renderMetricsChart(json.data);
      metricsError.style.display = 'none';
      metricsContainer.style.display = 'grid';
    } catch (err) {
      console.warn('Unable to load comparison CSV:', err);
      metricsContainer.style.display = 'none';
      metricsError.style.display = 'flex';
      metricsError.textContent = 'Model comparison results unavailable.';
    }
  }

  function renderMetricsTable(rows) {
    metricsTableBody.innerHTML = '';

    rows.forEach(item => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${escapeHtml(item.model)}</strong></td>
        <td title="${item.accuracy}">${(item.accuracy * 100).toFixed(2)}%</td>
        <td title="${item.precision}">${(item.precision * 100).toFixed(2)}%</td>
        <td title="${item.recall}">${(item.recall * 100).toFixed(2)}%</td>
        <td title="${item.f1_score}">${(item.f1_score * 100).toFixed(2)}%</td>
      `;
      metricsTableBody.appendChild(tr);
    });
  }

  function renderMetricsChart(rows) {
    if (typeof Chart === 'undefined') {
      console.warn('Chart.js library is not available.');
      return;
    }

    const canvas = document.getElementById('performanceChart');
    if (!canvas) return;

    if (chartInstance) {
      chartInstance.destroy();
    }

    const labels = rows.map(r => r.model);
    const accuracyData = rows.map(r => Number((r.accuracy * 100).toFixed(2)));
    const precisionData = rows.map(r => Number((r.precision * 100).toFixed(2)));
    const recallData = rows.map(r => Number((r.recall * 100).toFixed(2)));
    const f1Data = rows.map(r => Number((r.f1_score * 100).toFixed(2)));

    chartInstance = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Accuracy (%)',
            data: accuracyData,
            backgroundColor: '#2563eb',
            borderRadius: 3
          },
          {
            label: 'Precision (%)',
            data: precisionData,
            backgroundColor: '#059669',
            borderRadius: 3
          },
          {
            label: 'Recall (%)',
            data: recallData,
            backgroundColor: '#d97706',
            borderRadius: 3
          },
          {
            label: 'F1-Score (%)',
            data: f1Data,
            backgroundColor: '#7c3aed',
            borderRadius: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: 100,
            ticks: {
              stepSize: 20,
              callback: value => value + '%',
              font: {
                family: "'Inter', sans-serif",
                size: 11
              }
            },
            grid: {
              color: '#f1f5f9'
            }
          },
          x: {
            grid: {
              display: false
            },
            ticks: {
              font: {
                family: "'Inter', sans-serif",
                weight: '600',
                size: 12
              }
            }
          }
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              boxWidth: 12,
              padding: 12,
              font: {
                family: "'Inter', sans-serif",
                size: 11
              }
            }
          },
          tooltip: {
            backgroundColor: '#0f172a',
            padding: 10,
            callbacks: {
              label: context => ` ${context.dataset.label}: ${context.raw}%`
            }
          }
        }
      }
    });
  }

  loadMetrics();

  // ==========================================================================
  // 7. Quick Test Samples (Text Mode)
  // ==========================================================================
  const samples = {
    real: {
      title: "U.S. House passes defense policy bill with broad bipartisan support",
      text: "WASHINGTON (Reuters) - The U.S. House of Representatives on Tuesday overwhelmingly passed a comprehensive defense policy bill authorizing billions in national security spending, sending the measure to the Senate for approval.",
      model: "logistic_regression"
    },
    fake: {
      title: "BREAKING VIDEO: Secret Meeting Exposes Huge Government Scheme",
      text: "Watch this shocking footage via wire image! Insiders reveal unprecedented conspiracy that mainstream media refuses to report. Even top officials were caught off guard in leaked recording.",
      model: "logistic_regression"
    },
    political: {
      title: "White House announces official trade delegation schedule",
      text: "WASHINGTON (Reuters) - The White House announced on Wednesday that a senior diplomatic team will travel to Europe next week for scheduled bilateral discussions regarding international tariffs.",
      model: "svm"
    }
  };

  btnSampleReal.addEventListener('click', () => populateSample(samples.real));
  btnSampleFake.addEventListener('click', () => populateSample(samples.fake));
  btnSamplePolitical.addEventListener('click', () => populateSample(samples.political));

  function populateSample(sample) {
    switchTabMode('text');
    newsTitleInput.value = sample.title;
    newsTextInput.value = sample.text;
    modelSelector.value = sample.model;
    modelDescription.textContent = MODEL_DESCRIPTIONS[sample.model] || '';
    updateTextCounters(sample.title, sample.text, charWordCounter);
    hideError();
    resultSection.style.display = 'none';
  }

  // ==========================================================================
  // 8. Clear Button
  // ==========================================================================
  btnClear.addEventListener('click', () => {
    // Clear Text Mode
    newsTitleInput.value = '';
    newsTextInput.value = '';
    updateTextCounters('', '', charWordCounter);

    // Clear Document Mode
    currentFile = null;
    docFileInput.value = '';
    docExtractedTitle.value = '';
    docExtractedText.value = '';
    docPreviewBox.style.display = 'none';
    docDropzone.style.display = 'block';

    // Clear URL Mode
    newsUrlInput.value = '';
    urlExtractedTitle.value = '';
    urlExtractedText.value = '';
    urlPreviewBox.style.display = 'none';
    currentDomain = '';

    // Reset Model & Results
    modelSelector.value = 'logistic_regression';
    modelDescription.textContent = MODEL_DESCRIPTIONS.logistic_regression;
    hideError();
    resultSection.style.display = 'none';

    if (currentMode === 'text') newsTitleInput.focus();
    else if (currentMode === 'url') newsUrlInput.focus();
  });

  // ==========================================================================
  // 9. Keyboard Shortcut: Ctrl + Enter
  // ==========================================================================
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      const activeEl = document.activeElement;
      if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA')) {
        e.preventDefault();
        newsForm.requestSubmit ? newsForm.requestSubmit() : executePrediction();
      }
    }
  });

  // ==========================================================================
  // 10. Form Submission & Analysis
  // ==========================================================================
  newsForm.addEventListener('submit', (e) => {
    e.preventDefault();
    executePrediction();
  });

  async function executePrediction() {
    hideError();

    let title = '';
    let text = '';
    let sourceLabel = 'Direct Text Input';
    const modelVal = modelSelector.value;

    if (currentMode === 'text') {
      title = newsTitleInput.value.trim();
      text = newsTextInput.value.trim();
      sourceLabel = 'Direct Text Input';

      if (!title && !text) {
        showError('Please enter a news headline or article text before analyzing.');
        newsTitleInput.focus();
        return;
      }
    } else if (currentMode === 'document') {
      title = docExtractedTitle.value.trim();
      text = docExtractedText.value.trim();
      const fn = currentFile ? currentFile.name : 'Document';
      const ft = docBadgeType.textContent || 'DOC';
      sourceLabel = `Document (${fn} • ${ft})`;

      if (!title && !text) {
        showError('Please upload a PDF, Word, or text document and extract content before analyzing.');
        return;
      }
    } else if (currentMode === 'url') {
      title = urlExtractedTitle.value.trim();
      text = urlExtractedText.value.trim();
      const domain = currentDomain || (newsUrlInput.value ? 'Web Source' : '');
      sourceLabel = `Website Link (${domain})`;

      if (!title && !text) {
        // If URL is entered but not fetched yet, fetch automatically!
        if (newsUrlInput.value.trim()) {
          await fetchArticleFromUrl();
          title = urlExtractedTitle.value.trim();
          text = urlExtractedText.value.trim();
          if (!title && !text) return;
        } else {
          showError('Please enter a news website URL and click "Fetch Article" before analyzing.');
          newsUrlInput.focus();
          return;
        }
      }
    }

    // Set Loading State
    btnAnalyze.disabled = true;
    analyzeSpinner.style.display = 'inline-block';
    analyzeBtnText.textContent = 'Analyzing...';

    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: title,
          text: text,
          model: modelVal
        })
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'Failed to analyze text.');
      }

      displayResult(result, sourceLabel);
    } catch (err) {
      showError(err.message || 'Error communicating with machine learning backend.');
      resultSection.style.display = 'none';
    } finally {
      btnAnalyze.disabled = false;
      analyzeSpinner.style.display = 'none';
      analyzeBtnText.textContent = 'Analyze News';
    }
  }

  // ==========================================================================
  // 11. Result Display
  // ==========================================================================
  function displayResult(result, sourceLabel) {
    const isReal = result.prediction.toUpperCase() === 'REAL NEWS';

    resultVerdict.textContent = result.prediction;
    resultModelName.textContent = result.model;
    resultModelBadge.textContent = result.model;
    if (result.verification_detail) {
      resultSourceBadge.textContent = `Source: ${sourceLabel} • ${result.verification_detail}`;
    } else {
      resultSourceBadge.textContent = `Source: ${sourceLabel}`;
    }
    resultTokens.textContent = `${result.token_count || 0} tokens`;
    resultRawClass.textContent = `Class ${result.raw_class} (${isReal ? 'Real' : 'Fake'})`;

    if (isReal) {
      resultCard.className = 'card result-card is-real';
      if (result.verification_detail) {
        resultExplanationText.textContent =
          `Authenticated via ${result.verification_detail}. The trained ML pipeline verified authentic journalistic reporting standards, accredited news source attribution, and formal news syntax.`;
      } else {
        resultExplanationText.textContent =
          'The trained ML model identified formal journalistic syntax, structured news attribution (such as wire agency identifiers), and vocabulary distributions characteristic of verified reporting.';
      }
    } else {
      resultCard.className = 'card result-card is-fake';
      resultExplanationText.textContent =
        'The trained ML model identified sensationalist markers, informal phrasing, and lexical distributions characteristic of unverified, conspiracy, or fake news content.';
    }

    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // ==========================================================================
  // 12. Error Utilities
  // ==========================================================================
  function showError(msg) {
    formErrorText.textContent = msg;
    formError.style.display = 'flex';
  }

  function hideError() {
    formErrorText.textContent = '';
    formError.style.display = 'none';
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
});
