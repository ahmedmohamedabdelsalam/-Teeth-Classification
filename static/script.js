// DOM Elements
const input = document.getElementById('image-input');
const dropZone = document.getElementById('drop-zone');
const preview = document.getElementById('preview');
const previewContainer = document.getElementById('preview-container');
const uploadPrompt = document.getElementById('upload-prompt');
const predictBtn = document.getElementById('predict-btn');
const initialState = document.getElementById('initial-state');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const resultClass = document.getElementById('result-class');
const resultConfidence = document.getElementById('result-confidence');
const confidenceBar = document.getElementById('confidence-bar');
const scoreBars = document.getElementById('score-bars');
const statusText = document.getElementById('status-text');
const statusIndicator = document.getElementById('status-indicator');

let selectedFile = null;

// File Selection Handlers
dropZone.addEventListener('click', () => {
    console.log('Drop zone clicked, triggering input selection...');
    input.click();
});

input.addEventListener('change', (e) => {
    console.log('File selection changed, files count:', e.target.files.length);
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

// Drag & Drop Handlers
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
    }, false);
});

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
        dropZone.classList.add('border-blue-500', 'bg-blue-50/20');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
        dropZone.classList.remove('border-blue-500', 'bg-blue-50/20');
    }, false);
});

dropZone.addEventListener('drop', (e) => {
    console.log('Files dropped into drop zone');
    const files = e.dataTransfer.files;
    if (files.length) {
        handleFile(files[0]);
    }
}, false);

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Unauthorized file format. Please provide a standard clinical image (JPEG/PNG).');
        return;
    }
    
    selectedFile = file;
    const reader = new FileReader();
    
    reader.onload = (e) => {
        preview.src = e.target.result;
        previewContainer.classList.remove('hidden');
        uploadPrompt.classList.add('hidden');
        predictBtn.disabled = false;
        
        // UI Reset
        results.classList.add('hidden');
        initialState.classList.add('hidden'); // Hide "Waiting" text immediately
        loading.classList.add('hidden');
        
        statusText.innerText = 'Ready for Diagnostic Scan';
        statusIndicator.className = 'h-2 w-2 rounded-full bg-blue-500 pulse-light';
        
        console.log('File ingested successfully:', file.name);
    };
    
    reader.onerror = () => {
        alert('Failed to read image file.');
        console.error('File reader error');
    };
    
    reader.readAsDataURL(file);
}

// Prediction Trigger
predictBtn.onclick = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    // Enter processing state
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    predictBtn.disabled = true;
    
    statusText.innerText = 'Analyzing Scan...';
    statusIndicator.className = 'h-2 w-2 rounded-full bg-amber-500 animate-pulse';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Clinical Scan Failure');
        }

        const data = await response.json();
        
        // Simulate high-fidelity analysis delay
        setTimeout(() => {
            renderAnalysis(data);
            statusText.innerText = 'Analysis Complete';
            statusIndicator.className = 'h-2 w-2 rounded-full bg-emerald-500';
            predictBtn.disabled = false;
        }, 1500);

    } catch (err) {
        console.error('System Fault:', err);
        statusText.innerText = 'System Fault Detected';
        statusIndicator.className = 'h-2 w-2 rounded-full bg-rose-500';
        alert('Diagnostic Error: ' + err.message);
        predictBtn.disabled = false;
        loading.classList.add('hidden');
    }
};

function renderAnalysis(data) {
    loading.classList.add('hidden');
    results.classList.remove('hidden');
    
    // Core metrics
    resultClass.innerText = data.prediction;
    const confidencePercent = (data.confidence * 100).toFixed(1);
    resultConfidence.innerText = `${confidencePercent}%`;
    confidenceBar.style.width = `${confidencePercent}%`;
    
    // Differential distribution
    scoreBars.innerHTML = '';
    
    // Sort scores by probability
    const sortedScores = Object.entries(data.all_scores).sort(([,a], [,b]) => b - a);
    
    sortedScores.forEach(([label, score]) => {
        const pct = (score * 100).toFixed(1);
        const bar = document.createElement('div');
        bar.className = 'group';
        bar.innerHTML = `
            <div class="flex items-center justify-between mb-2">
                <span class="text-[11px] font-bold text-slate-700 uppercase tracking-tighter">${label}</span>
                <span class="text-[11px] font-bold text-slate-400 group-hover:text-blue-600 transition-colors">${pct}%</span>
            </div>
            <div class="h-1 bg-slate-100 rounded-full overflow-hidden">
                <div style="width: 0%" class="progress-bar-inner h-full bg-slate-800 rounded-full group-hover:bg-blue-600 transition-all duration-700"></div>
            </div>
        `;
        scoreBars.appendChild(bar);
        
        // Trigger animation
        setTimeout(() => {
            const inner = bar.querySelector('.progress-bar-inner');
            if (inner) inner.style.width = `${pct}%`;
        }, 50);
    });
}
