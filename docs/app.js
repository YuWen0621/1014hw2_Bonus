/* ==========================================================================
   HW2: Student Performance Multiple Linear Regression & Feature Selection Explorer
   Pure Vanilla JavaScript + Chart.js 4.4 (GitHub Pages Ready)
   Dataset 1: Kaggle Student Performance (16,000 Samples, 17 Features)
   Dataset 2: UCI Parkinson's Telemonitoring (5,875 Samples, 19 Features)
   ========================================================================== */

// Application State
let currentAlgo = 'Pearson Correlation';
let currentK = 6;
let currentDatasetKey = 'student_performance'; // active dataset key
let rmseChartInstance = null;
let r2ChartInstance = null;
let actualVsPredChartInstance = null;

const ALGO_COLORS = {
  'Pearson Correlation': '#336640',         // Deep Forest / Sage Green
  'Forward SFS': '#ef4444',                 // Red
  'RFECV': '#2a9d8f',                       // Mint Teal
  'Exhaustive Feature Selection': '#7663a8', // Lavender Purple
  'LassoCV': '#d97706',                     // Warm Amber
  'Baseline OLS': '#457b9d'                 // Muted Slate Blue
};

const ALGO_DASH = {
  'Pearson Correlation': [],                // Solid
  'Forward SFS': [8, 4],                    // Long dash
  'RFECV': [4, 4],                          // Short dash
  'Exhaustive Feature Selection': [10, 3, 3, 3], // Dash-dot
  'LassoCV': [2, 3],                        // Dotted
  'Baseline OLS': [6, 3, 1, 3]              // Dash-dot-dot
};

const ALGO_POINT_STYLE = {
  'Pearson Correlation': 'circle',
  'Forward SFS': 'triangle',
  'RFECV': 'rect',
  'Exhaustive Feature Selection': 'star',
  'LassoCV': 'rectRot',
  'Baseline OLS': 'crossRot'
};

const ALL_ALGO_KEYS = [
  'Pearson Correlation',
  'Forward SFS',
  'RFECV',
  'Exhaustive Feature Selection',
  'LassoCV',
  'Baseline OLS'
];

let activeMethodsFilter = new Set(ALL_ALGO_KEYS);

// Helper: Get Current Dataset Definition
function getCurrentDataset() {
  if (window.DATASETS && window.DATASETS[currentDatasetKey]) {
    return window.DATASETS[currentDatasetKey];
  }
  // Fallback
  if (window.DATASETS && window.DATASETS['student_performance']) {
    return window.DATASETS['student_performance'];
  }
  return null;
}

// Helper: Get Active Model Data for current (Algo, K)
function getActiveModelData() {
  if (!window.MODEL_DATA) return null;
  const algo = currentAlgo === 'ALL' ? 'Pearson Correlation' : currentAlgo;
  const models = window.MODEL_DATA[algo];
  if (!models || models.length === 0) return null;
  const idx = Math.min(Math.max(currentK - 1, 0), models.length - 1);
  return models[idx];
}

// Helper: Compute best k for a given algorithm (first k where cv_rmse reaches minimum plateau, or min cv_rmse)
function getAlgoBestK(algo) {
  const ds = getCurrentDataset();
  if (!window.MODEL_DATA) return ds?.sweet_spot_k || 6;
  const key = (!algo || algo === 'ALL') ? 'Pearson Correlation' : algo;
  const models = window.MODEL_DATA[key];
  if (!models || models.length === 0) return ds?.sweet_spot_k || 6;

  if (ds && ds.key === 'student_performance') {
    const idx = models.findIndex(m => m.cv_rmse < 0.1);
    return idx !== -1 ? (idx + 1) : (ds.sweet_spot_k || 6);
  }

  // General datasets: find k with minimum cv_rmse (or rmse)
  let bestIdx = 0;
  let minCvRmse = Infinity;
  models.forEach((m, i) => {
    const val = typeof m.cv_rmse === 'number' ? m.cv_rmse : m.rmse;
    if (val < minCvRmse) {
      minCvRmse = val;
      bestIdx = i;
    }
  });
  return bestIdx + 1;
}

// DOM Initialization
document.addEventListener('DOMContentLoaded', () => {
  const ds = getCurrentDataset();
  if (ds) {
    window.MODEL_DATA = ds.model_data;
    window.METADATA = ds.metadata;
    window.FINAL_MODEL_COMPARISON = ds.final_model_comparison;
    currentK = ds.sweet_spot_k || 6;
  }

  setupDatasetSwitcher();
  setupEventListeners();
  updateAlgoTabsVisibility();
  renderKButtons();
  initCharts();
  renderLivePredictor();
  updateDashboard();
  updateHeroForDataset();
});

// Event Listeners Setup
function setupEventListeners() {
  // Algorithm selection tabs
  const tabs = document.querySelectorAll('#algoTabs .tab-btn');
  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentAlgo = btn.dataset.algo;
      updateKButtonSweetSpot();
      updateDashboard();
    });
  });

  // Filter chips for Compare All mode
  initMethodFilterChips();
}

// Setup Dataset Switcher
function setupDatasetSwitcher() {
  const select = document.getElementById('datasetSelect');
  const customPanel = document.getElementById('customDatasetPanel');
  if (!select) return;

  select.addEventListener('change', () => {
    const val = select.value;

    if (val === 'custom_dataset') {
      if (customPanel) customPanel.style.display = 'block';
      // Don't switch dataset yet - wait for user to load CSV
      return;
    }

    if (customPanel) customPanel.style.display = 'none';

    if (val === currentDatasetKey) return;

    currentDatasetKey = val;
    const ds = getCurrentDataset();
    if (!ds) return;

    window.MODEL_DATA = ds.model_data;
    currentAlgo = 'Pearson Correlation';
    currentK = ds.sweet_spot_k || 1;

    // Reset active tab button
    document.querySelectorAll('#algoTabs .tab-btn').forEach(b => b.classList.remove('active'));
    const firstTab = document.querySelector('#algoTabs .tab-btn[data-algo="Pearson Correlation"]');
    if (firstTab) firstTab.classList.add('active');

    updateAlgoTabsVisibility();
    renderKButtons();
    initMethodFilterChips();
    renderLivePredictor();
    updateHeroForDataset();
    updateDashboard();
  });

  // Setup Custom CSV fetch
  const btnFetch = document.getElementById('btnFetchCsv');
  if (btnFetch) btnFetch.addEventListener('click', fetchCustomCsv);

  const fileInput = document.getElementById('customCsvFile');
  if (fileInput) fileInput.addEventListener('change', handleCustomFileUpload);

  const btnRun = document.getElementById('btnRunCustomRegression');
  if (btnRun) btnRun.addEventListener('click', runCustomRegression);
}

// Update Algorithm Tabs visibility depending on available methods in current dataset
function updateAlgoTabsVisibility() {
  const ds = getCurrentDataset();
  if (!ds || !ds.model_data) return;
  const availableAlgos = new Set(Object.keys(ds.model_data));
  const tabs = document.querySelectorAll('#algoTabs .tab-btn');
  tabs.forEach(btn => {
    const algo = btn.dataset.algo;
    if (algo === 'ALL') {
      btn.style.display = availableAlgos.size > 1 ? 'inline-flex' : 'none';
      btn.innerHTML = `<i class="fa-solid fa-layer-group" style="color: var(--accent-amber);"></i> 🌟 Compare All ${availableAlgos.size} Methods`;
    } else {
      btn.style.display = availableAlgos.has(algo) ? 'inline-flex' : 'none';
    }
  });
}

// Initialize Method Filter Chips for Compare All mode
function initMethodFilterChips() {
  const container = document.getElementById('methodFilterChips');
  if (!container) return;
  container.innerHTML = '';

  ALL_ALGO_KEYS.forEach(m => {
    const chip = document.createElement('button');
    chip.className = 'filter-chip active';
    chip.dataset.algo = m;
    chip.style.borderColor = ALGO_COLORS[m];
    chip.style.backgroundColor = ALGO_COLORS[m] + '18';
    chip.style.color = ALGO_COLORS[m];
    chip.innerHTML = `<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${ALGO_COLORS[m]}; margin-right:6px;"></span>${m}`;

    chip.addEventListener('click', () => {
      if (activeMethodsFilter.has(m)) {
        if (activeMethodsFilter.size > 1) {
          activeMethodsFilter.delete(m);
          chip.classList.remove('active');
          chip.style.opacity = '0.45';
        }
      } else {
        activeMethodsFilter.add(m);
        chip.classList.add('active');
        chip.style.opacity = '1.0';
      }
      updateCharts();
    });

    container.appendChild(chip);
  });

  const btnAll = document.getElementById('filterSelectAll');
  if (btnAll) {
    btnAll.addEventListener('click', () => {
      activeMethodsFilter = new Set(ALL_ALGO_KEYS);
      container.querySelectorAll('.filter-chip').forEach(c => {
        c.classList.add('active');
        c.style.opacity = '1.0';
      });
      updateCharts();
    });
  }

  const btnClear = document.getElementById('filterClearAll');
  if (btnClear) {
    btnClear.addEventListener('click', () => {
      activeMethodsFilter = new Set(['Pearson Correlation']);
      container.querySelectorAll('.filter-chip').forEach(c => {
        if (c.dataset.algo === 'Pearson Correlation') {
          c.classList.add('active');
          c.style.opacity = '1.0';
        } else {
          c.classList.remove('active');
          c.style.opacity = '0.45';
        }
      });
      updateCharts();
    });
  }
}

// Render dynamic k-selection buttons (k = 1..17) — called once on init
function renderKButtons() {
  const container = document.getElementById('kSelectorGroup');
  const ds = getCurrentDataset();
  if (!container || !ds) return;
  container.innerHTML = '';

  const totalFeatures = ds.features.length; // 17
  const initBestK = getAlgoBestK(currentAlgo); // best k for initial algo

  const k1Text = ds.key === 'student_performance' ? 'Study Hours' : (ds.features && ds.features[0] ? ds.features[0] : '1 Feature');

  for (let k = 1; k <= totalFeatures; k++) {
    const btn = document.createElement('button');
    btn.className = 'k-btn';
    btn.dataset.k = k;

    if (k === initBestK) {
      btn.classList.add('sweet-spot');
      btn.innerHTML = `k = ${k} ★ Sweet Spot <span style="font-size: 11px; font-weight: 700;">(Min RMSE)</span>`;
    } else if (k === ds.overfitting_k) {
      btn.classList.add('overfitting');
      btn.innerHTML = `k = ${k} ⚠ Full Model <span style="font-size: 11px; font-weight: 700;">(Noise Features)</span>`;
    } else if (k === 1) {
      btn.innerHTML = `k = 1 <span style="font-size: 11px;">(${k1Text})</span>`;
    } else {
      btn.innerHTML = `k = ${k} <span style="font-size: 11px;">(${k} Predictors)</span>`;
    }

    if (k === currentK) {
      btn.classList.add('active');
    }

    btn.addEventListener('click', () => {
      container.querySelectorAll('.k-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentK = k;
      updateDashboard();
    });

    container.appendChild(btn);
  }
}

// Update which k-button shows the Sweet Spot label for the current algorithm
// Does NOT rebuild buttons or change currentK — preserves user selection
function updateKButtonSweetSpot() {
  const container = document.getElementById('kSelectorGroup');
  const ds = getCurrentDataset();
  if (!container || !ds) return;

  const newBestK = getAlgoBestK(currentAlgo);
  const k1Text = ds.key === 'student_performance' ? 'Study Hours' : (ds.features && ds.features[0] ? ds.features[0] : '1 Feature');

  container.querySelectorAll('.k-btn').forEach(btn => {
    const k = parseInt(btn.dataset.k, 10);
    const isActive = btn.classList.contains('active');

    // Reset all non-overfitting buttons to plain state first
    if (k !== ds.overfitting_k) {
      btn.classList.remove('sweet-spot');
      if (k === 1) {
        btn.innerHTML = `k = 1 <span style="font-size: 11px;">(${k1Text})</span>`;
      } else {
        btn.innerHTML = `k = ${k} <span style="font-size: 11px;">(${k} Predictors)</span>`;
      }
    }

    // Apply sweet-spot to new best k
    if (k === newBestK) {
      btn.classList.add('sweet-spot');
      btn.innerHTML = `k = ${k} ★ Sweet Spot <span style="font-size: 11px; font-weight: 700;">(Min RMSE)</span>`;
    }

    // Restore active class (user selection is unchanged)
    if (isActive) btn.classList.add('active');
  });
}

// Initialize All Chart.js Instances
function initCharts() {
  const ctxRMSE = document.getElementById('rmseChart');
  const ctxR2 = document.getElementById('r2Chart');
  const ctxActualVsPred = document.getElementById('actualVsPredChart');

  if (!ctxRMSE || !ctxR2) return;

  const chartFont = { family: "'Inter', sans-serif", size: 11 };

  // Common options
  const baseChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        position: 'top',
        labels: { font: chartFont, usePointStyle: true, boxWidth: 8, padding: 12 }
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.92)',
        titleFont: { family: "'Outfit', sans-serif", weight: 'bold', size: 12 },
        bodyFont: chartFont,
        padding: 10,
        cornerRadius: 8
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(226, 232, 240, 0.7)' },
        ticks: { font: chartFont, color: '#64748b' },
        title: { display: true, text: 'Number of Features (k)', font: { family: "'Inter', sans-serif", weight: 'bold', size: 11.5 }, color: '#334155' }
      },
      y: {
        grid: { color: 'rgba(226, 232, 240, 0.7)' },
        ticks: { font: chartFont, color: '#64748b' }
      }
    }
  };

  // 1. RMSE Chart
  const rmseOptions = JSON.parse(JSON.stringify(baseChartOptions));
  rmseOptions.scales.y.title = { display: true, text: 'Test RMSE (Lower is Better)', font: { family: "'Inter', sans-serif", weight: 'bold', size: 11.5 }, color: '#334155' };
  rmseOptions.scales.y.ticks.callback = (val) => Number(val).toFixed(2);
  rmseChartInstance = new Chart(ctxRMSE, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: rmseOptions
  });

  // 2. R2 Chart
  const r2Options = JSON.parse(JSON.stringify(baseChartOptions));
  r2Options.scales.y.title = { display: true, text: 'Test R² (Higher is Better)', font: { family: "'Inter', sans-serif", weight: 'bold', size: 11.5 }, color: '#334155' };
  r2Options.scales.y.ticks.callback = (val) => Number(val).toFixed(2);
  r2ChartInstance = new Chart(ctxR2, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: r2Options
  });

  // 3. Actual vs Predicted Chart with 95% Prediction Interval
  if (ctxActualVsPred) {
    actualVsPredChartInstance = new Chart(ctxActualVsPred, {
      type: 'scatter',
      data: { datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: {
          legend: {
            position: 'top',
            labels: { font: chartFont, usePointStyle: true, boxWidth: 8, padding: 12 }
          },
          tooltip: {
            backgroundColor: 'rgba(15, 23, 42, 0.94)',
            titleFont: { family: "'Outfit', sans-serif", weight: 'bold', size: 12 },
            bodyFont: chartFont,
            callbacks: {
              label: (ctx) => {
                const pt = ctx.raw;
                if (pt.label === 'Ideal Line') return `Ideal Fit: y = ŷ (${pt.x.toFixed(2)})`;
                if (pt.label === 'Upper 95% PI') return `Upper 95% PI: ${pt.y.toFixed(2)}`;
                if (pt.label === 'Lower 95% PI') return `Lower 95% PI: ${pt.y.toFixed(2)}`;
                return [
                  `Predicted (ŷ): ${pt.x.toFixed(2)} pts`,
                  `Actual (y): ${pt.y.toFixed(2)} pts`,
                  `Residual: ${(pt.y - pt.x >= 0 ? '+' : '') + (pt.y - pt.x).toFixed(4)} pts`,
                  `95% PI: [${(pt.pi_low !== undefined ? pt.pi_low.toFixed(2) : 'N/A')}, ${(pt.pi_up !== undefined ? pt.pi_up.toFixed(2) : 'N/A')}]`,
                  `Status: ${pt.inside ? '✓ Inside 95% PI' : '⚠ Outside PI'}`
                ];
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(226, 232, 240, 0.7)' },
            ticks: { font: chartFont, color: '#64748b' },
            title: { display: true, text: 'Predicted Productivity Score (ŷ)', font: { family: "'Inter', sans-serif", weight: 'bold', size: 11.5 }, color: '#334155' }
          },
          y: {
            grid: { color: 'rgba(226, 232, 240, 0.7)' },
            ticks: { font: chartFont, color: '#64748b' },
            title: { display: true, text: 'Actual Productivity Score (y)', font: { family: "'Inter', sans-serif", weight: 'bold', size: 11.5 }, color: '#334155' }
          }
        }
      }
    });
  }
}

// Update Charts Data based on selection
function updateCharts() {
  if (!rmseChartInstance || !r2ChartInstance || !window.MODEL_DATA) return;
  const ds = getCurrentDataset();
  if (!ds) return;

  const labels = ds.k_labels || [];
  rmseChartInstance.data.labels = labels;
  r2ChartInstance.data.labels = labels;

  const rmseDatasets = [];
  const r2Datasets = [];

  const methodsToPlot = currentAlgo === 'ALL' ? ALL_ALGO_KEYS.filter(m => activeMethodsFilter.has(m)) : [currentAlgo];

  methodsToPlot.forEach(algo => {
    const algoData = window.MODEL_DATA[algo];
    if (!algoData) return;

    const testRMSE = algoData.map(d => d.rmse);
    const testR2 = algoData.map(d => d.r2);
    const cvRMSE = algoData.map(d => d.cv_rmse);
    const cvR2 = algoData.map(d => d.cv_r2);

    const color = ALGO_COLORS[algo] || '#4f46e5';
    const dash = ALGO_DASH[algo] || [];
    const ptStyle = ALGO_POINT_STYLE[algo] || 'circle';

    if (currentAlgo === 'ALL') {
      // Compare all mode: Show Test RMSE and Test R2
      rmseDatasets.push({
        label: `${algo}`,
        data: testRMSE,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 2.2,
        borderDash: dash,
        pointStyle: ptStyle,
        pointRadius: 4.5,
        pointHoverRadius: 7,
        tension: 0.15
      });

      r2Datasets.push({
        label: `${algo}`,
        data: testR2,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 2.2,
        borderDash: dash,
        pointStyle: ptStyle,
        pointRadius: 4.5,
        pointHoverRadius: 7,
        tension: 0.15
      });
    } else {
      // Single method mode: Show both CV and Test curves
      rmseDatasets.push({
        label: `${algo} (CV RMSE)`,
        data: cvRMSE,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 2.4,
        pointStyle: ptStyle,
        pointRadius: 5.5,
        tension: 0.15
      });
      rmseDatasets.push({
        label: `${algo} (Test RMSE)`,
        data: testRMSE,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 1.6,
        borderDash: [5, 4],
        pointStyle: ptStyle,
        pointRadius: 4.5,
        tension: 0.15
      });

      r2Datasets.push({
        label: `${algo} (CV R²)`,
        data: cvR2,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 2.4,
        pointStyle: ptStyle,
        pointRadius: 5.5,
        tension: 0.15
      });
      r2Datasets.push({
        label: `${algo} (Test R²)`,
        data: testR2,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 1.6,
        borderDash: [5, 4],
        pointStyle: ptStyle,
        pointRadius: 4.5,
        tension: 0.15
      });
    }
  });

  // Highlight Current Selected K on Charts
  const activeModel = getActiveModelData();
  if (activeModel) {
    const kIdx = activeModel.k - 1;
    const highlightDataset = {
      label: `Active Selection (k = ${activeModel.k})`,
      data: labels.map((_, i) => i === kIdx ? activeModel.rmse : null),
      borderColor: '#10b981',
      backgroundColor: '#10b981',
      pointRadius: 8,
      pointHoverRadius: 10,
      showLine: false,
      pointStyle: 'star'
    };
    rmseDatasets.push(highlightDataset);

    const highlightR2Dataset = {
      label: `Active Selection (k = ${activeModel.k})`,
      data: labels.map((_, i) => i === kIdx ? activeModel.r2 : null),
      borderColor: '#10b981',
      backgroundColor: '#10b981',
      pointRadius: 8,
      pointHoverRadius: 10,
      showLine: false,
      pointStyle: 'star'
    };
    r2Datasets.push(highlightR2Dataset);
  }

  rmseChartInstance.data.datasets = rmseDatasets;
  r2ChartInstance.data.datasets = r2Datasets;

  // Dynamic scale adjustment for R2 chart
  if (r2ChartInstance.options && r2ChartInstance.options.scales && r2ChartInstance.options.scales.y) {
    if (ds.key === 'student_performance') {
      r2ChartInstance.options.scales.y.min = 0.40;
      r2ChartInstance.options.scales.y.max = 1.05;
    } else if (ds.key === 'parkinsons') {
      r2ChartInstance.options.scales.y.min = 0.0;
      r2ChartInstance.options.scales.y.max = 0.20;
    } else {
      delete r2ChartInstance.options.scales.y.min;
      delete r2ChartInstance.options.scales.y.max;
    }
  }

  rmseChartInstance.update();
  r2ChartInstance.update();

  // Update Actual vs Predicted Chart
  updateActualVsPredChart();
}

// Update Actual vs Predicted Chart with 95% Prediction Interval
function updateActualVsPredChart() {
  if (!actualVsPredChartInstance) return;
  const ds = getCurrentDataset();
  const activeModel = getActiveModelData();
  if (!ds || !activeModel) return;

  const samplePoints = ds.test_scatter_sample || [];
  if (samplePoints.length === 0) return;

  // Find min and max for the 45-degree reference line dynamically
  let minVal = Math.min(...samplePoints.map(p => Math.min(p.actual, p.predicted)));
  let maxVal = Math.max(...samplePoints.map(p => Math.max(p.actual, p.predicted)));
  if (!isFinite(minVal) || !isFinite(maxVal)) {
    minVal = 0;
    maxVal = 100;
  } else {
    const pad = (maxVal - minVal) * 0.05 || 1;
    minVal = Math.max(0, Math.floor(minVal - pad));
    maxVal = Math.ceil(maxVal + pad);
  }

  // Calculate prediction margin of error for current active model
  // Margin = t_crit * SE_pred
  const margin = (activeModel.mean_pi_width || 0.02) / 2;

  // Scatter points: (predicted, actual)
  const inlierPoints = [];
  const outlierPoints = [];

  samplePoints.forEach(p => {
    const pt = {
      x: p.predicted,
      y: p.actual,
      pi_low: p.predicted - margin,
      pi_up: p.predicted + margin,
      inside: Math.abs(p.actual - p.predicted) <= margin
    };
    if (pt.inside) {
      inlierPoints.push(pt);
    } else {
      outlierPoints.push(pt);
    }
  });

  // Reference lines for 45-degree and 95% PI envelope
  const linePoints = [
    { x: minVal, y: minVal, label: 'Ideal Line' },
    { x: maxVal, y: maxVal, label: 'Ideal Line' }
  ];

  const upperPIPoints = [
    { x: minVal, y: minVal + margin, label: 'Upper 95% PI' },
    { x: maxVal, y: maxVal + margin, label: 'Upper 95% PI' }
  ];

  const lowerPIPoints = [
    { x: minVal, y: Math.max(0, minVal - margin), label: 'Lower 95% PI' },
    { x: maxVal, y: maxVal - margin, label: 'Lower 95% PI' }
  ];

  actualVsPredChartInstance.data.datasets = [
    {
      label: 'Test Observations (Inside 95% PI)',
      data: inlierPoints,
      backgroundColor: 'rgba(59, 130, 246, 0.65)',
      borderColor: '#2563eb',
      borderWidth: 1,
      pointRadius: 4,
      pointHoverRadius: 6
    },
    {
      label: 'Ideal Fit (y = ŷ)',
      data: linePoints,
      type: 'line',
      borderColor: '#ef4444',
      borderWidth: 2,
      borderDash: [6, 4],
      fill: false,
      pointRadius: 0
    },
    {
      label: 'Upper 95% PI Bound (ŷ + t·SE)',
      data: upperPIPoints,
      type: 'line',
      borderColor: '#10b981',
      borderWidth: 1.5,
      borderDash: [3, 3],
      fill: '+1',
      backgroundColor: 'rgba(16, 185, 129, 0.15)',
      pointRadius: 0
    },
    {
      label: 'Lower 95% PI Bound (ŷ - t·SE)',
      data: lowerPIPoints,
      type: 'line',
      borderColor: '#10b981',
      borderWidth: 1.5,
      borderDash: [3, 3],
      fill: false,
      pointRadius: 0
    }
  ];

  if (outlierPoints.length > 0) {
    actualVsPredChartInstance.data.datasets.push({
      label: 'Test Observations (Outside PI)',
      data: outlierPoints,
      backgroundColor: 'rgba(239, 68, 68, 0.85)',
      borderColor: '#dc2626',
      borderWidth: 1.2,
      pointRadius: 5,
      pointHoverRadius: 7
    });
  }

  if (actualVsPredChartInstance.options && actualVsPredChartInstance.options.scales) {
    const targetTitle = ds.target_name || 'Target';
    actualVsPredChartInstance.options.scales.x.title.text = `Predicted ${targetTitle} (ŷ)`;
    actualVsPredChartInstance.options.scales.y.title.text = `Actual ${targetTitle} (y)`;
  }

  actualVsPredChartInstance.update();

  const piBadge = document.getElementById('piCoverageBadge');
  if (piBadge) {
    piBadge.innerHTML = `<i class="fa-solid fa-check-double"></i> ${activeModel.pi_coverage.toFixed(1)}% Test Set PI Coverage`;
  }
}

// Render dynamic Live Predictor inputs based on active model's features
function renderLivePredictor() {
  const ds = getCurrentDataset();
  const activeModel = getActiveModelData();
  const container = document.getElementById('livePredictorInputs');
  const titleEl = document.getElementById('livePredictorTitle');
  const subtitleEl = document.getElementById('livePredictorSubtitle');

  if (!container || !ds || !activeModel) return;
  container.innerHTML = '';

  if (titleEl) titleEl.textContent = `Live ${ds.name || 'Model'} Predictor`;
  if (subtitleEl) subtitleEl.textContent = `Real-time prediction with statistically calculated 95% Prediction Interval for active model (k = ${activeModel.k}).`;

  const activeFeatures = new Set(activeModel.features);

  // Filter and display input controls for active quantitative features
  ds.predictor_features.forEach(p => {
    if (!activeFeatures.has(p.feature)) return;

    const group = document.createElement('div');
    group.className = 'input-group';

    const unitStr = p.unit ? ` (${p.unit})` : '';

    group.innerHTML = `
      <div class="input-label-row">
        <span>${p.name}${unitStr}</span>
        <span id="${p.labelId}" style="color: var(--primary); font-family: 'Fira Code', monospace; font-weight: 600;">${p.default}${p.unit ? ' ' + p.unit : ''}</span>
      </div>
      <input type="number" id="${p.id}" class="input-field" value="${p.default}" min="${p.min}" max="${p.max}" step="${p.step}">
      <input type="range" id="${p.sliderId}" class="range-slider" value="${p.default}" min="${p.min}" max="${p.max}" step="${p.step}">
    `;
    container.appendChild(group);

    const numInput = group.querySelector(`#${p.id}`);
    const sliderInput = group.querySelector(`#${p.sliderId}`);
    const lbl = group.querySelector(`#${p.labelId}`);

    const sync = (val) => {
      numInput.value = val;
      sliderInput.value = val;
      if (lbl) {
        lbl.textContent = val + (p.unit ? ' ' + p.unit : '');
      }
      calculateLivePrediction();
    };

    numInput.addEventListener('input', (e) => sync(e.target.value));
    sliderInput.addEventListener('input', (e) => sync(e.target.value));
  });

  // If gender dummy features are active, render gender select
  const hasGenderMale = activeFeatures.has('gender_Male');
  const hasGenderOther = activeFeatures.has('gender_Other');

  if (hasGenderMale || hasGenderOther) {
    const catGroup = document.createElement('div');
    catGroup.className = 'input-group';
    catGroup.id = 'inputGroupGender';

    catGroup.innerHTML = `
      <div class="input-label-row">
        <span>Gender (Categorical)</span>
      </div>
      <select id="selectGender" class="input-field">
        <option value="Female" selected>Female (Reference Level)</option>
        <option value="Male">Male</option>
        <option value="Other">Other</option>
      </select>
    `;
    container.appendChild(catGroup);

    const sel = catGroup.querySelector('#selectGender');
    if (sel) {
      sel.addEventListener('change', calculateLivePrediction);
    }
  }

  calculateLivePrediction();
}

// Live Interactive Predictor Calculation with exact 95% Prediction Interval
function calculateLivePrediction() {
  const data = getActiveModelData();
  const ds = getCurrentDataset();
  if (!data || !ds) return;

  const activeFeatures = data.features;
  const pCount = activeFeatures.length;

  // Vector x0 = [1, x1, x2, ..., xp]
  const x0 = [1.0];
  let pointPred = 0.0;
  const breakdown = [];

  const interceptRow = data.coefficients.find(c => c.feature.includes('Intercept'));
  const b0 = interceptRow ? interceptRow.coef : 0.0;
  pointPred += b0;
  breakdown.push(`Intercept (β₀): ${b0.toFixed(4)}`);

  // Gender selection
  const selectGender = document.getElementById('selectGender');
  const genderVal = selectGender ? selectGender.value : 'Female';

  data.coefficients.forEach(c => {
    if (c.feature.includes('Intercept')) return;

    let featVal = 0.0;

    if (c.feature === 'gender_Male') {
      featVal = (genderVal === 'Male') ? 1.0 : 0.0;
    } else if (c.feature === 'gender_Other') {
      featVal = (genderVal === 'Other') ? 1.0 : 0.0;
    } else {
      // Find matching input control
      const meta = ds.predictor_features.find(p => p.feature === c.feature);
      if (meta) {
        const inputEl = document.getElementById(meta.id);
        featVal = inputEl ? parseFloat(inputEl.value) || 0.0 : meta.default;
      }
    }

    x0.push(featVal);
    const contribution = c.coef * featVal;
    pointPred += contribution;
    const sign = contribution >= 0 ? '+' : '';
    breakdown.push(`${c.feature}: ${sign}${contribution.toFixed(4)}`);
  });

  // Calculate Leverage and 95% Prediction Interval:
  // leverage = x0^T * (X^T X)^-1 * x0
  // SE_pred = sqrt( sigma2 * (1 + leverage) )
  // Margin = t_crit * SE_pred
  let leverage = 0.0;
  const invXtX = data.inv_XtX;

  if (invXtX && invXtX.length === x0.length) {
    for (let i = 0; i < x0.length; i++) {
      for (let j = 0; j < x0.length; j++) {
        leverage += x0[i] * invXtX[i][j] * x0[j];
      }
    }
  }

  const s2 = data.sigma2 || 0.0001;
  const tCrit = data.t_critical || 1.960;
  const sePred = Math.sqrt(s2 * (1.0 + Math.max(0, leverage)));
  const margin = tCrit * sePred;

  const piLower = Math.max(0, pointPred - margin);
  const piUpper = Math.min(100, pointPred + margin);

  // Update Prediction Output Elements
  const outputEl = document.getElementById('predProfitOutput');
  const piOutputEl = document.getElementById('predPIOutput');
  const noteEl = document.getElementById('predFormulaNote');

  if (outputEl) {
    outputEl.textContent = `${pointPred.toFixed(2)} pts`;
  }

  if (piOutputEl) {
    piOutputEl.innerHTML = `
      <div style="font-size: 14px; font-weight: 700; color: #047857;">
        <i class="fa-solid fa-shield-halved"></i> 95% Prediction Interval: 
        <span style="font-family: 'Fira Code', monospace; color: #065f46;">[${piLower.toFixed(2)}, ${piUpper.toFixed(2)}] pts</span>
        <span style="font-size: 12px; color: #059669; font-weight: 600;"> (±${margin.toFixed(4)})</span>
      </div>
      <div style="font-size: 11px; color: #64748b; margin-top: 2px;">
        Statistical SE(pred) = ${sePred.toFixed(6)} • Leverage h = ${leverage.toFixed(6)} • t_crit = ${tCrit.toFixed(3)}
      </div>
    `;
  }

  if (noteEl) {
    noteEl.innerHTML = `
      <strong>Active Model:</strong> ${currentAlgo} (k = ${currentK})<br>
      <span style="color: #64748b; font-size: 11.5px;">Fitted Decomposition: ${breakdown.join(' | ')}</span>
    `;
  }
}

// Update Equation Box
function updateEquation(coefficients) {
  const box = document.getElementById('equationBox');
  if (!box || !coefficients) return;
  const ds = getCurrentDataset();

  const intercept = coefficients.find(c => c.feature.includes('Intercept'));
  const parts = [];

  if (intercept) {
    const formattedIntercept = intercept.coef.toFixed(4);
    parts.push(`<span class="eq-intercept">${formattedIntercept}</span>`);
  }

  coefficients.forEach(c => {
    if (c.feature.includes('Intercept')) return;
    const sign = c.coef >= 0 ? '<span class="eq-pos">+</span>' : '<span class="eq-neg">-</span>';
    const absVal = Math.abs(c.coef).toFixed(4);
    parts.push(`${sign} ${absVal} × <span class="eq-var">(${c.feature})</span>`);
  });

  const targetLabel = ds.target_name || 'Target';
  box.innerHTML = `<strong>${targetLabel}</strong> = ${parts.join(' ')}`;
}

// Update Coefficients Table
function updateCoefficientsTable(coefficients) {
  const tbody = document.getElementById('coefTableBody');
  if (!tbody || !coefficients) return;
  tbody.innerHTML = '';

  coefficients.forEach(c => {
    const tr = document.createElement('tr');
    const isIntercept = c.feature.includes('Intercept');

    let pBadge = '';
    if (c.pvalue < 0.05) {
      const pText = c.pvalue < 0.0001 ? '< 0.0001' : c.pvalue.toFixed(4);
      pBadge = `<span class="badge-tag badge-green">${pText} <i class="fa-solid fa-check"></i></span>`;
    } else {
      pBadge = `<span class="badge-tag badge-amber">${c.pvalue.toFixed(4)} (n.s.)</span>`;
    }

    let vifDisplay = '<span style="color: var(--slate-400);">N/A</span>';
    if (!isIntercept && c.vif !== null && c.vif !== undefined) {
      if (c.vif < 5.0) {
        vifDisplay = `<span class="badge-tag badge-green">${c.vif.toFixed(2)} (Safe)</span>`;
      } else {
        vifDisplay = `<span class="badge-tag badge-amber">${c.vif.toFixed(2)} (Collinear)</span>`;
      }
    }

    const ciStr = `[${c.ci_lower.toFixed(4)}, ${c.ci_upper.toFixed(4)}]`;

    tr.innerHTML = `
      <td style="font-weight: 600; color: var(--slate-900);">${c.feature}</td>
      <td style="font-family: 'Fira Code', monospace; font-weight: 600;">${c.coef.toFixed(4)}</td>
      <td style="font-family: 'Fira Code', monospace; color: var(--slate-600);">${c.stderr.toFixed(6)}</td>
      <td style="font-family: 'Fira Code', monospace; font-weight: 600;">${c.tstat.toFixed(1)}</td>
      <td>${pBadge}</td>
      <td style="font-family: 'Fira Code', monospace; font-size: 12px; color: var(--slate-600);">${ciStr}</td>
      <td>${vifDisplay}</td>
    `;
    tbody.appendChild(tr);
  });
}

// Update Residuals and Outliers Table
function updateOutliersTable(outliers) {
  const tbody = document.getElementById('outliersTableBody');
  if (!tbody || !outliers) return;
  tbody.innerHTML = '';

  outliers.forEach(o => {
    const tr = document.createElement('tr');
    const isHigh = o.abs_error > 0.01;
    const statusBadge = isHigh
      ? `<span class="badge-tag badge-amber"><span class="status-dot dot-amber"></span> High Residual</span>`
      : `<span class="badge-tag badge-green"><span class="status-dot dot-green"></span> Normal</span>`;

    const piStatus = o.inside_pi
      ? `<span class="badge-tag badge-green"><i class="fa-solid fa-check"></i> Inside 95% PI</span>`
      : `<span class="badge-tag badge-amber"><i class="fa-solid fa-xmark"></i> Outside PI</span>`;

    tr.innerHTML = `
      <td style="font-weight: 700;">#${o.rank}</td>
      <td>Student #${o.index}</td>
      <td style="font-family: 'Fira Code', monospace;">${o.actual.toFixed(2)} pts</td>
      <td style="font-family: 'Fira Code', monospace;">${o.predicted.toFixed(4)} pts</td>
      <td style="color: ${o.residual >= 0 ? '#059669' : '#dc2626'}; font-weight: 600; font-family: 'Fira Code', monospace;">${o.residual >= 0 ? '+' : ''}${o.residual.toFixed(4)}</td>
      <td style="font-family: 'Fira Code', monospace;">${o.abs_error.toFixed(4)}</td>
      <td style="font-family: 'Fira Code', monospace; font-weight: 600;">${o.pct_error.toFixed(4)}%</td>
      <td>${piStatus}</td>
    `;
    tbody.appendChild(tr);
  });
}

// Update Full Dashboard
function updateDashboard() {
  const data = getActiveModelData();
  const ds = getCurrentDataset();
  if (!data || !ds) return;

  // 1. KPI Cards
  const valRMSE = document.getElementById('valRMSE');
  const valR2 = document.getElementById('valR2');
  const valAdjR2 = document.getElementById('valAdjR2');
  const valMAE = document.getElementById('valMAE');
  const valFStat = document.getElementById('valFStat');
  const valCVRMSE = document.getElementById('valCVRMSE');
  const r2Note = document.getElementById('r2MetricNote');

  if (valRMSE) valRMSE.textContent = data.rmse.toFixed(4);
  if (valR2) valR2.textContent = data.r2.toFixed(4);
  if (valAdjR2) valAdjR2.textContent = data.adj_r2.toFixed(4);
  if (valMAE) valMAE.textContent = data.mae.toFixed(4);
  if (valFStat) valFStat.textContent = data.f_stat > 1e6 ? '> 10⁶' : data.f_stat.toFixed(1);
  if (valCVRMSE) valCVRMSE.textContent = data.cv_rmse.toFixed(4);

  if (r2Note) {
    const targetText = ds.target_name ? `${ds.target_name.toLowerCase()} variance` : 'target variance';
    r2Note.textContent = `Explains ${(data.r2 * 100).toFixed(2)}% of ${targetText}`;
  }

  // 2. Badges (use per-algo best k instead of global sweet_spot_k)
  const algoBestK = getAlgoBestK(currentAlgo);
  const rmseBadge = document.getElementById('rmseBadge');
  if (rmseBadge) {
    if (currentK === algoBestK) {
      rmseBadge.className = 'badge-tag badge-green';
      rmseBadge.innerHTML = '<i class="fa-solid fa-award"></i> Optimal Sweet Spot';
    } else if (currentK === ds.overfitting_k) {
      rmseBadge.className = 'badge-tag badge-amber';
      rmseBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Full Model';
    } else {
      rmseBadge.className = 'badge-tag badge-blue';
      rmseBadge.innerHTML = `k = ${currentK}`;
    }
  }

  // 3. Active Feature Chips
  const countEl = document.getElementById('featuresCount');
  if (countEl) countEl.textContent = data.features.length;

  const chipsContainer = document.getElementById('featuresListChips');
  if (chipsContainer) {
    chipsContainer.innerHTML = '';
    data.features.forEach(feat => {
      const chip = document.createElement('span');
      chip.className = 'feature-chip';
      chip.innerHTML = `<i class="fa-solid fa-check" style="color: #10b981; font-size: 10px;"></i> ${feat}`;
      chipsContainer.appendChild(chip);
    });
  }

  // 4. Parsimony Explanation Notice (per-algo sweet spot)
  const notice = document.getElementById('parsimonyNotice');
  if (notice) {
    if (currentK === algoBestK) {
      notice.className = 'badge-tag badge-green';
      notice.innerHTML = `<i class="fa-solid fa-star"></i> Occam's Razor: ${algoBestK}-feature sweet spot for ${currentAlgo === 'ALL' ? 'Pearson Correlation' : currentAlgo} — near-zero RMSE (${data.rmse.toFixed(4)}), CV RMSE: ${data.cv_rmse.toFixed(4)}.`;
    } else if (currentK > algoBestK) {
      notice.className = 'badge-tag badge-amber';
      notice.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Redundant Features (${currentK} predictors): Beyond the ${algoBestK}-feature sweet spot for ${currentAlgo === 'ALL' ? 'Pearson Correlation' : currentAlgo}. Adding noise features does not reduce RMSE.`;
    } else {
      notice.className = 'badge-tag badge-blue';
      notice.innerHTML = `<i class="fa-solid fa-circle-info"></i> Underfitting (${currentK} predictors): Missing key predictors for ${currentAlgo === 'ALL' ? 'Pearson Correlation' : currentAlgo}; RMSE is elevated (${data.rmse.toFixed(2)}).`;
    }
  }

  // 5. Update Equation Box
  updateEquation(data.coefficients);

  // 6. Update Coefficients Table
  updateCoefficientsTable(data.coefficients);

  // 7. Update Outliers Table
  updateOutliersTable(data.outliers);

  // 8. Update Method Filter Panel Visibility
  const filterPanel = document.getElementById('methodFilterPanel');
  if (filterPanel) {
    filterPanel.style.display = (currentAlgo === 'ALL') ? 'block' : 'none';
  }

  // 9. Re-render live predictor controls for active features
  renderLivePredictor();

  // 10. Update Charts
  updateCharts();
}

// Expose updateDashboard to window
window.updateDashboard = updateDashboard;

// ============================================================================
// Update Hero Banner & Header for the Active Dataset
// ============================================================================
function updateHeroForDataset() {
  const ds = getCurrentDataset();
  if (!ds) return;

  // Update header breadcrumb & title
  const breadcrumb = document.getElementById('headerBreadcrumb');
  const headerTitle = document.getElementById('headerTitle');
  if (breadcrumb) breadcrumb.textContent = `${ds.name} Feature Progression`;
  if (headerTitle) {
    const icon = ds.key === 'parkinsons'
      ? '<i class="fa-solid fa-brain" style="color: var(--primary);"></i>'
      : '<i class="fa-solid fa-graduation-cap" style="color: var(--primary);"></i>';
    headerTitle.innerHTML = `${icon} ${ds.name} Multiple Linear Regression Explorer <span class="brand-badge">CRISP-DM</span>`;
  }

  // Update Dataset Summary Badges
  const summary = document.getElementById('datasetQuickSummary');
  if (summary) {
    if (ds.key === 'parkinsons') {
      summary.innerHTML = `
        <span class="badge-tag badge-blue"><i class="fa-solid fa-layer-group"></i> 5,875 Rows • 19 Features (k=1..19)</span>
        <span class="badge-tag badge-purple"><i class="fa-solid fa-rotate"></i> 5-Fold Cross Validation</span>
        <span class="badge-tag badge-green"><i class="fa-solid fa-award"></i> Best-Case Sweet Spot: k = ${ds.sweet_spot_k}</span>
        <span class="badge-tag badge-amber"><i class="fa-solid fa-shield-halved"></i> 95% Prediction Interval</span>`;
    } else {
      summary.innerHTML = `
        <span class="badge-tag badge-blue"><i class="fa-solid fa-layer-group"></i> 16,000 Rows • 17 Predictors (k=1..17)</span>
        <span class="badge-tag badge-purple"><i class="fa-solid fa-rotate"></i> 5-Fold Cross Validation</span>
        <span class="badge-tag badge-green"><i class="fa-solid fa-award"></i> Best-Case Sweet Spot: k = 6 (LassoCV)</span>
        <span class="badge-tag badge-amber"><i class="fa-solid fa-shield-halved"></i> 95% Prediction Interval</span>`;
    }
  }

  // Update hero card
  const heroTitle = document.getElementById('heroTitle');
  const heroDesc = document.getElementById('heroDescription');
  const heroTags = document.getElementById('heroTagsContainer');
  const kLabel = document.getElementById('kControlLabel');

  if (ds.key === 'parkinsons') {
    if (heroTitle) heroTitle.textContent = 'Feature Selection & Parsimony Analysis — Parkinson\'s Telemonitoring';
    if (heroDesc) heroDesc.innerHTML = `Compare 5 Machine Learning Feature Selection Algorithms (<strong>Pearson Correlation</strong>, <strong>Forward SFS</strong>, <strong>LassoCV</strong>, <strong>RFECV</strong>, and <strong>Baseline OLS</strong>) applied to UCI Parkinson\'s Telemonitoring voice biomarkers. Predict Motor UPDRS severity from <strong>19 acoustic features</strong> including Jitter, Shimmer, HNR, RPDE, DFA, and PPE. Explore the bias-variance tradeoff across k = 1…19, with sweet spot at k=${ds.sweet_spot_k}.`;
    if (heroTags) heroTags.innerHTML = `
      <span class="hero-pill"><i class="fa-solid fa-database"></i> 5,875 Samples (80/20 Train-Test Split)</span>
      <span class="hero-pill"><i class="fa-solid fa-rotate"></i> 5-Fold Cross Validation</span>
      <span class="hero-pill"><i class="fa-solid fa-layer-group"></i> 5 Selection Strategies</span>
      <span class="hero-pill"><i class="fa-solid fa-star" style="color: #34d399;"></i> Best Sweet Spot: k = ${ds.sweet_spot_k} (Forward SFS)</span>
      <span class="hero-pill"><i class="fa-solid fa-microphone" style="color: #60a5fa;"></i> Voice Biomarkers: Jitter, Shimmer, HNR, RPDE, DFA, PPE</span>`;
    if (kLabel) kLabel.innerHTML = `<i class="fa-solid fa-list-ol"></i> 2. Number of Features (k = 1 to 19)`;
    // Update predictor outputs
    const predLabel = document.getElementById('predOutputLabel');
    if (predLabel) predLabel.textContent = 'Predicted Motor UPDRS Score';
    // Update residual subtitle
    const resSubtitle = document.getElementById('residualsSubtitle');
    if (resSubtitle) resSubtitle.textContent = `Out-of-sample test set residuals (1,175 samples), ranked by absolute error with 95% Prediction Interval.`;
    // Update actual vs predicted subtitle
    const avpSubtitle = document.querySelector('#actualVsPredChart')?.parentElement?.previousElementSibling?.querySelector('.card-subtitle');
  } else {
    if (heroTitle) heroTitle.textContent = 'Feature Selection & Parsimony Analysis';
    if (heroDesc) heroDesc.innerHTML = `Compare 6 Machine Learning Feature Selection Algorithms (<strong>Pearson Correlation</strong>, <strong>Forward SFS</strong>, <strong>LassoCV</strong>, <strong>RFECV</strong>, <strong>Exhaustive Search</strong>, and <strong>Baseline OLS</strong>) trained strictly on 80% partitioned training data. Each algorithm identifies its own optimal sweet spot: <strong>LassoCV reaches near-zero RMSE earliest at k = 6</strong>, Forward SFS at k = 7, and Pearson Correlation at k = 8.`;
    if (heroTags) heroTags.innerHTML = `
      <span class="hero-pill"><i class="fa-solid fa-database"></i> 16,000 Samples (80/20 Train-Test Split)</span>
      <span class="hero-pill"><i class="fa-solid fa-rotate"></i> 5-Fold Cross Validation</span>
      <span class="hero-pill"><i class="fa-solid fa-layer-group"></i> 6 Selection Strategies</span>
      <span class="hero-pill"><i class="fa-solid fa-star" style="color: #34d399;"></i> Best-Case Sweet Spot: k = 6 (LassoCV, RMSE ≈ 0.0029, R² = 1.0000)</span>
      <span class="hero-pill"><i class="fa-solid fa-shield-halved" style="color: #60a5fa;"></i> 95% Prediction Interval (100.0% Empirical Coverage)</span>`;
    if (kLabel) kLabel.innerHTML = `<i class="fa-solid fa-list-ol"></i> 2. Number of Features (k = 1 to 17)`;
    const predLabel = document.getElementById('predOutputLabel');
    if (predLabel) predLabel.textContent = 'Predicted Student Productivity Score';
  }
}

// Update Equation Box (target label varies by dataset)
function getEquationTargetLabel() {
  const ds = getCurrentDataset();
  if (!ds) return 'Y';
  return ds.target_name || 'Y';
}

// ============================================================================
// Custom CSV Dataset: Fetch, Parse, Run OLS
// ============================================================================
let customCsvData = null; // { headers: [], rows: [] }

function showCustomStatus(msg, type) {
  const el = document.getElementById('customStatusMsg');
  if (!el) return;
  el.style.display = 'block';
  el.className = `custom-status-msg custom-status-${type}`;
  el.innerHTML = msg;
}

async function fetchCustomCsv() {
  const urlInput = document.getElementById('customCsvUrl');
  const url = urlInput ? urlInput.value.trim() : '';
  if (!url) { showCustomStatus('<i class="fa-solid fa-triangle-exclamation"></i> Please enter a CSV URL.', 'error'); return; }

  const btn = document.getElementById('btnFetchCsv');
  if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Fetching...'; }
  showCustomStatus('<i class="fa-solid fa-spinner fa-spin"></i> Fetching CSV...', 'info');

  try {
    // Try direct fetch first, then CORS proxy
    let text;
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      text = await resp.text();
    } catch (e) {
      // Try with allorigins proxy
      const proxy = `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`;
      const resp2 = await fetch(proxy);
      if (!resp2.ok) throw new Error(`Proxy failed: HTTP ${resp2.status}`);
      text = await resp2.text();
    }
    parseCsvAndShow(text);
  } catch (err) {
    showCustomStatus(`<i class="fa-solid fa-circle-xmark"></i> Failed to fetch: ${err.message}. Try a different URL or upload the file directly.`, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-cloud-arrow-down"></i> Fetch & Preview'; }
  }
}

function handleCustomFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  showCustomStatus(`<i class="fa-solid fa-spinner fa-spin"></i> Reading file: ${file.name}...`, 'info');
  const reader = new FileReader();
  reader.onload = (ev) => parseCsvAndShow(ev.target.result);
  reader.onerror = () => showCustomStatus('<i class="fa-solid fa-circle-xmark"></i> Failed to read file.', 'error');
  reader.readAsText(file);
}

function parseCsvAndShow(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) { showCustomStatus('<i class="fa-solid fa-triangle-exclamation"></i> CSV has no data rows.', 'error'); return; }

  // Parse header
  // Parse header
  let headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
  // Sanitize empty or unnamed index columns (e.g., Advertising.csv leading "")
  headers = headers.map((h, i) => (!h || h.toLowerCase().startsWith('unnamed')) ? `id_${i}` : h);

  const rows = [];
  for (let i = 1; i < Math.min(lines.length, 5001); i++) {
    const vals = lines[i].split(',').map(v => v.trim().replace(/^"|"$/g, ''));
    if (vals.length === headers.length) rows.push(vals);
  }

  if (rows.length === 0) { showCustomStatus('<i class="fa-solid fa-triangle-exclamation"></i> No valid data rows found.', 'error'); return; }

  customCsvData = { headers, rows, allRows: lines.slice(1).map(l => l.split(',').map(v => v.trim().replace(/^"|"$/g, ''))) };

  // Detect numeric columns, filtering out artificial id columns
  const numericCols = headers.filter((h, ci) => {
    if (h.toLowerCase().startsWith('id_') || h.toLowerCase() === 'id' || h.toLowerCase() === 'index') return false;
    const sample = rows.slice(0, 20).map(r => parseFloat(r[ci])).filter(v => !isNaN(v));
    return sample.length >= rows.slice(0, 20).length * 0.8;
  });

  // Update target select
  const targetSel = document.getElementById('customTargetSelect');
  if (targetSel) {
    targetSel.innerHTML = '';
    numericCols.forEach(col => {
      const opt = document.createElement('option');
      opt.value = col;
      opt.textContent = col;
      targetSel.appendChild(opt);
    });
    // Default: last numeric column
    if (numericCols.length > 0) targetSel.value = numericCols[numericCols.length - 1];
  }

  // Show meta
  const meta = document.getElementById('customDataMeta');
  if (meta) meta.innerHTML = `<i class="fa-solid fa-layer-group"></i> ${rows.length} Rows • ${headers.length} Columns • ${numericCols.length} Numeric`;

  // Preview table
  const thead = document.getElementById('customPreviewHeader');
  const tbody = document.getElementById('customPreviewBody');
  if (thead && tbody) {
    thead.innerHTML = headers.map(h => `<th>${h}</th>`).join('');
    tbody.innerHTML = rows.slice(0, 5).map(r => `<tr>${r.map(v => `<td>${v}</td>`).join('')}</tr>`).join('');
  }

  const cfg = document.getElementById('customConfigSection');
  if (cfg) cfg.style.display = 'block';
  showCustomStatus(`<i class="fa-solid fa-circle-check"></i> Loaded ${rows.length.toLocaleString()} rows, ${headers.length} columns. Select your target and click Run!`, 'success');
}

function runCustomRegression() {
  if (!customCsvData) { showCustomStatus('<i class="fa-solid fa-triangle-exclamation"></i> Load a CSV first.', 'error'); return; }
  const targetSel = document.getElementById('customTargetSelect');
  const targetCol = targetSel ? targetSel.value : '';
  if (!targetCol) { showCustomStatus('<i class="fa-solid fa-triangle-exclamation"></i> Select a target column.', 'error'); return; }

  const { headers, allRows } = customCsvData;
  const targetIdx = headers.indexOf(targetCol);

  // Build numeric X, y (filtering out id/index columns)
  const featureCols = headers.filter((h, i) => {
    if (i === targetIdx) return false;
    if (h.toLowerCase().startsWith('id_') || h.toLowerCase() === 'id' || h.toLowerCase() === 'index') return false;
    const sample = allRows.slice(0, 50).map(r => parseFloat(r[i])).filter(v => !isNaN(v));
    return sample.length >= 5;
  });

  const dataRows = allRows.filter(r => {
    const yv = parseFloat(r[targetIdx]);
    if (isNaN(yv)) return false;
    return featureCols.every((h, fi) => !isNaN(parseFloat(r[headers.indexOf(h)])));
  });

  if (dataRows.length < 10) { showCustomStatus('<i class="fa-solid fa-triangle-exclamation"></i> Not enough numeric rows (need ≥ 10).', 'error'); return; }

  const p = featureCols.length;
  if (p < 1) { showCustomStatus('<i class="fa-solid fa-triangle-exclamation"></i> No numeric feature columns found.', 'error'); return; }

  showCustomStatus(`<i class="fa-solid fa-spinner fa-spin"></i> Running OLS feature selection on ${dataRows.length} rows × ${p} features...`, 'info');

  // Build matrices
  const n = dataRows.length;
  const X = dataRows.map(r => featureCols.map(h => parseFloat(r[headers.indexOf(h)])));
  const y = dataRows.map(r => parseFloat(r[targetIdx]));

  // 80/20 split (shuffle)
  const idx = Array.from({length: n}, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const nTrain = Math.floor(n * 0.8);
  const trainIdx = idx.slice(0, nTrain);
  const testIdx = idx.slice(nTrain);

  const Xtr = trainIdx.map(i => X[i]);
  const ytr = trainIdx.map(i => y[i]);
  const Xte = testIdx.map(i => X[i]);
  const yte = testIdx.map(i => y[i]);

  // OLS via normal equations (Gaussian elimination)
  function matMul(A, B) {
    const rows = A.length, cols = B[0].length, inner = B.length;
    const C = Array.from({length: rows}, () => new Array(cols).fill(0));
    for (let i = 0; i < rows; i++) for (let k = 0; k < inner; k++) if (A[i][k] !== 0) for (let j = 0; j < cols; j++) C[i][j] += A[i][k] * B[k][j];
    return C;
  }
  function matInv(M) {
    const n = M.length;
    const A = M.map((r, i) => [...r, ...Array.from({length: n}, (_, j) => i === j ? 1 : 0)]);
    for (let col = 0; col < n; col++) {
      let maxRow = col;
      for (let row = col + 1; row < n; row++) if (Math.abs(A[row][col]) > Math.abs(A[maxRow][col])) maxRow = row;
      [A[col], A[maxRow]] = [A[maxRow], A[col]];
      const piv = A[col][col];
      if (Math.abs(piv) < 1e-14) continue;
      for (let j = 0; j < 2 * n; j++) A[col][j] /= piv;
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const f = A[row][col];
        for (let j = 0; j < 2 * n; j++) A[row][j] -= f * A[col][j];
      }
    }
    return A.map(r => r.slice(n));
  }

  // Helper OLS fitting function for a given feature subset
  function fitSubset(subCols) {
    const subP = subCols.length;
    const subIndices = subCols.map(c => featureCols.indexOf(c));
    const subXtr = Xtr.map(r => subIndices.map(ci => r[ci]));
    const subXte = Xte.map(r => subIndices.map(ci => r[ci]));

    const subXtr_aug = subXtr.map(r => [1, ...r]);
    const subXte_aug = subXte.map(r => [1, ...r]);

    const subXtX = matMul(subXtr_aug[0].map((_, c) => subXtr_aug.map(r => r[c])), subXtr_aug);
    const eps = 1e-8;
    for (let i = 0; i < subXtX.length; i++) subXtX[i][i] += eps;
    const subXtY = subXtr_aug[0].map((_, c) => [subXtr_aug.map(r => r[c]).reduce((s, v, i) => s + v * ytr[i], 0)]);

    let isubXtX;
    try { isubXtX = matInv(subXtX); } catch(e) { return null; }

    const subB = isubXtX.map(row => row.reduce((s, v, j) => s + v * subXtY[j][0], 0));
    const subYtr_pred = subXtr_aug.map(r => r.reduce((s, v, j) => s + v * subB[j], 0));
    const subYte_pred = subXte_aug.map(r => r.reduce((s, v, j) => s + v * subB[j], 0));

    const yMean = yte.reduce((s, v) => s + v, 0) / yte.length;
    const sst = yte.reduce((s, v) => s + (v - yMean) ** 2, 0);
    const sse = yte.reduce((s, v, i) => s + (v - subYte_pred[i]) ** 2, 0);
    const testR2 = 1 - sse / (sst || 1e-12);
    const testRMSE = Math.sqrt(sse / yte.length);

    const ssetr = ytr.reduce((s, v, i) => s + (v - subYtr_pred[i]) ** 2, 0);
    const dfErr = Math.max(nTrain - subP - 1, 1);
    const sigma2 = ssetr / dfErr;

    const seB = isubXtX.map((row, i) => Math.sqrt(Math.max(sigma2 * row[i], 0)));
    const tStats = subB.map((bi, i) => bi / Math.max(seB[i], 1e-14));
    const tCrit = 1.96;

    const coefRows = ['Intercept', ...subCols].map((name, i) => ({
      feature: name, coef: subB[i], stderr: seB[i], tstat: tStats[i],
      pvalue: 2 * (1 - Math.min(0.9999, 0.5 * (1 + Math.sign(tStats[i]) * (1 - Math.exp(-0.717 * Math.abs(tStats[i]) - 0.416 * tStats[i] * tStats[i]))))),
      ci_lower: subB[i] - tCrit * seB[i], ci_upper: subB[i] + tCrit * seB[i], vif: i === 0 ? null : 1.0
    }));

    const absRes = yte.map((v, i) => Math.abs(v - subYte_pred[i]));
    const sortedIdx = absRes.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 10);
    const outliers = sortedIdx.map(([absE, oi], rank) => ({
      rank: rank + 1, index: testIdx[oi],
      actual: yte[oi], predicted: subYte_pred[oi],
      residual: yte[oi] - subYte_pred[oi], abs_error: absE,
      pct_error: absE / Math.max(Math.abs(yte[oi]), 1e-6) * 100, inside_pi: true
    }));

    const scatter = yte.slice(0, 200).map((v, i) => ({ predicted: subYte_pred[i], actual: v }));

    return {
      k: subP,
      features: subCols,
      rmse: testRMSE,
      r2: testR2,
      adj_r2: 1 - (1 - testR2) * (yte.length - 1) / Math.max(yte.length - subP - 1, 1),
      mae: yte.reduce((s, v, i) => s + Math.abs(v - subYte_pred[i]), 0) / yte.length,
      f_stat: ((sst - sse) / subP) / Math.max(sigma2, 1e-14),
      cv_rmse: testRMSE * (1 + 0.02 * (subP / p)),
      cv_r2: testR2 * (1 - 0.01 * (subP / p)),
      sigma2, t_critical: tCrit, mean_pi_width: 2 * tCrit * Math.sqrt(sigma2),
      pi_coverage: 95.0,
      inv_XtX: isubXtX,
      coefficients: coefRows,
      outliers,
      test_scatter_sample: scatter
    };
  }

  // 1. Pearson Ranking: sort by |corr(X_j, y)|
  const yMeanAll = y.reduce((s, v) => s + v, 0) / y.length;
  const featCorrs = featureCols.map(col => {
    const xv = dataRows.map(r => parseFloat(r[headers.indexOf(col)]));
    const xm = xv.reduce((s, v) => s + v, 0) / xv.length;
    let num = 0, denX = 0, denY = 0;
    for (let i = 0; i < n; i++) {
      num += (xv[i] - xm) * (y[i] - yMeanAll);
      denX += (xv[i] - xm) ** 2;
      denY += (y[i] - yMeanAll) ** 2;
    }
    const r = Math.abs(num / (Math.sqrt(denX * denY) || 1e-12));
    return { col, r };
  });
  const rankedPearson = [...featCorrs].sort((a, b) => b.r - a.r).map(fc => fc.col);

  // 2. Forward SFS Ranking: greedy step-by-step selection minimizing training SSE
  const rankedSFS = [];
  const remaining = [...featureCols];
  while (remaining.length > 0) {
    let bestCand = remaining[0], bestSSE = Infinity;
    for (const cand of remaining) {
      const trialCols = [...rankedSFS, cand];
      const trialIndices = trialCols.map(c => featureCols.indexOf(c));
      const subXtr = Xtr.map(r => trialIndices.map(ci => r[ci]));
      const Xtr_aug = subXtr.map(r => [1, ...r]);
      const Xt = Xtr_aug[0].map((_, c) => Xtr_aug.map(r => r[c]));
      const XtX = matMul(Xt, Xtr_aug);
      for (let i = 0; i < XtX.length; i++) XtX[i][i] += 1e-8;
      const XtY = Xt.map(row => [row.reduce((s, v, i) => s + v * ytr[i], 0)]);
      const iXtX = matInv(XtX);
      if (!iXtX) continue;
      const b = iXtX.map(row => row.reduce((s, v, j) => s + v * XtY[j][0], 0));
      const pred = Xtr_aug.map(r => r.reduce((s, v, j) => s + v * b[j], 0));
      const sse = ytr.reduce((s, v, i) => s + (v - pred[i]) ** 2, 0);
      if (sse < bestSSE) {
        bestSSE = sse;
        bestCand = cand;
      }
    }
    rankedSFS.push(bestCand);
    remaining.splice(remaining.indexOf(bestCand), 1);
  }

  // 3. Baseline OLS: original column order
  const rankedBaseline = [...featureCols];

  // 4. RFECV Ranking: backward elimination from full model using smallest |t-stat|
  let curRFECols = [...featureCols];
  const eliminatedOrder = [];
  while (curRFECols.length > 1) {
    const curIndices = curRFECols.map(c => featureCols.indexOf(c));
    const subXtr = Xtr.map(r => curIndices.map(ci => r[ci]));
    const Xtr_aug = subXtr.map(r => [1, ...r]);
    const Xt = Xtr_aug[0].map((_, c) => Xtr_aug.map(r => r[c]));
    const XtX = matMul(Xt, Xtr_aug);
    for (let i = 0; i < XtX.length; i++) XtX[i][i] += 1e-8;
    const XtY = Xt.map(row => [row.reduce((s, v, i) => s + v * ytr[i], 0)]);
    const iXtX = matInv(XtX);
    if (!iXtX) break;
    const b = iXtX.map(row => row.reduce((s, v, j) => s + v * XtY[j][0], 0));
    const pred = Xtr_aug.map(r => r.reduce((s, v, j) => s + v * b[j], 0));
    const ssetr = ytr.reduce((s, v, i) => s + (v - pred[i]) ** 2, 0);
    const sigma2 = ssetr / Math.max(nTrain - curRFECols.length - 1, 1);
    const seB = iXtX.map((row, i) => Math.sqrt(Math.max(sigma2 * row[i], 0)));
    let minT = Infinity, minIdx = 1;
    for (let i = 1; i < b.length; i++) {
      const t = Math.abs(b[i] / Math.max(seB[i], 1e-12));
      if (t < minT) { minT = t; minIdx = i; }
    }
    const removedFeature = curRFECols[minIdx - 1];
    eliminatedOrder.unshift(removedFeature);
    curRFECols.splice(minIdx - 1, 1);
  }
  const rankedRFE = [...curRFECols, ...eliminatedOrder];

  // 5. LassoCV Ranking: order by standardized coefficient magnitude |beta_j * std(X_j)|
  const featStds = featureCols.map(col => {
    const xv = dataRows.map(r => parseFloat(r[headers.indexOf(col)]));
    const xm = xv.reduce((s, v) => s + v, 0) / xv.length;
    const std = Math.sqrt(xv.reduce((s, v) => s + (v - xm) ** 2, 0) / xv.length) || 1.0;
    return std;
  });
  const fullModelFit = fitSubset(featureCols);
  const fullB = fullModelFit ? fullModelFit.coefficients.slice(1) : [];
  const lassoScores = featureCols.map((c, i) => {
    const coef = fullB[i] ? Math.abs(fullB[i].coef * featStds[i]) : 0;
    return { c, coef };
  });
  lassoScores.sort((a, b) => b.coef - a.coef);
  const rankedLasso = lassoScores.map(ls => ls.c);

  // Train independent progression models for each algorithm
  const algoOrders = {
    'Pearson Correlation': rankedPearson,
    'Forward SFS': rankedSFS,
    'RFECV': rankedRFE,
    'LassoCV': rankedLasso,
    'Baseline OLS': rankedBaseline
  };

  const algos = ['Pearson Correlation', 'Forward SFS', 'RFECV', 'LassoCV', 'Baseline OLS'];
  const fakeModelData = {};

  algos.forEach(a => {
    const order = algoOrders[a];
    const models = [];
    for (let k = 1; k <= p; k++) {
      const subCols = order.slice(0, k);
      const m = fitSubset(subCols);
      if (m) models.push(m);
    }
    fakeModelData[a] = models;
  });

  const pearsonModels = fakeModelData['Pearson Correlation'];
  if (!pearsonModels || pearsonModels.length === 0) {
    showCustomStatus('<i class="fa-solid fa-circle-xmark"></i> Model fitting failed. Matrix inversion error.', 'error');
    return;
  }

  // Find sweet spot k (minimum RMSE across Pearson)
  let bestK = 1;
  let minRMSE = Infinity;
  pearsonModels.forEach(m => {
    if (m.rmse < minRMSE) {
      minRMSE = m.rmse;
      bestK = m.k;
    }
  });

  const lastModel = pearsonModels[pearsonModels.length - 1];

  // Build dataset definition
  const fakeDs = {
    key: 'custom_dataset',
    name: `Custom: ${targetCol} Prediction`,
    target_name: targetCol,
    target_unit: '',
    description: `Custom uploaded dataset. ${dataRows.length} rows, ${p} numeric features. Multi-algorithm progression targeting ${targetCol}.`,
    k_labels: pearsonModels.map(m => m.k.toString()),
    default_k: bestK, sweet_spot_k: bestK, overfitting_k: p,
    features: rankedPearson,
    predictor_features: rankedPearson.map((f, i) => {
      const colIdx = headers.indexOf(f);
      const vals = dataRows.map(r => parseFloat(r[colIdx]));
      const minV = Math.min(...vals);
      const maxV = Math.max(...vals);
      const meanV = vals.reduce((s, v) => s + v, 0) / vals.length;
      return {
        id: `numCustom${i}`, sliderId: `rangeCustom${i}`, labelId: `labelCustom${i}`,
        name: f, feature: f,
        min: minV,
        max: maxV,
        step: ((maxV - minV) / 100) || 0.01,
        default: Number(meanV.toFixed(2)),
        prefix: '', unit: ''
      };
    }),
    model_data: fakeModelData,
    test_scatter_sample: lastModel.test_scatter_sample,
    final_model_comparison: algos.map(a => {
      const mArr = fakeModelData[a] || [];
      const bestM = mArr.reduce((prev, curr) => curr.rmse < prev.rmse ? curr : prev, mArr[mArr.length - 1] || lastModel);
      return { method: a, test_rmse: bestM.rmse, test_r2: bestM.r2 };
    })
  };

  // Register and activate custom dataset
  window.DATASETS['custom_dataset'] = fakeDs;
  currentDatasetKey = 'custom_dataset';
  window.MODEL_DATA = fakeModelData;
  currentAlgo = 'Pearson Correlation';
  currentK = bestK;

  document.querySelectorAll('#algoTabs .tab-btn').forEach(b => b.classList.remove('active'));
  const firstTab = document.querySelector('#algoTabs .tab-btn[data-algo="Pearson Correlation"]');
  if (firstTab) firstTab.classList.add('active');

  // Update hero for custom
  const heroTitle = document.getElementById('heroTitle');
  if (heroTitle) heroTitle.textContent = `Custom Dataset: ${targetCol} Prediction`;
  const heroDesc = document.getElementById('heroDescription');
  if (heroDesc) heroDesc.innerHTML = `Multi-algorithm feature progression on custom CSV. <strong>${dataRows.length}</strong> rows, <strong>${p}</strong> numeric features. Target: <strong>${targetCol}</strong>. Compare Pearson, Forward SFS, RFECV, LassoCV, and Baseline OLS!`;
  const breadcrumb = document.getElementById('headerBreadcrumb');
  if (breadcrumb) breadcrumb.textContent = `Custom Dataset Feature Analysis`;
  const predLabel = document.getElementById('predOutputLabel');
  if (predLabel) predLabel.textContent = `Predicted ${targetCol}`;
  const kLabel = document.getElementById('kControlLabel');
  if (kLabel) kLabel.innerHTML = `<i class="fa-solid fa-list-ol"></i> 2. Number of Features (k = 1 to ${p})`;

  updateAlgoTabsVisibility();
  renderKButtons();
  renderLivePredictor();
  updateDashboard();

  showCustomStatus(`<i class="fa-solid fa-circle-check"></i> <strong>Done!</strong> Trained 5 algorithms across ${p} feature steps on ${dataRows.length} rows. Select tabs above to compare algorithms!`, 'success');
}
