// Content Integrity Platform - Frontend JavaScript

// API Configuration
const API_BASE_URL = window.location.origin;

// DOM Elements
const inputText = document.getElementById('inputText');
const charCount = document.getElementById('charCount');
const checkAI = document.getElementById('checkAI');
const checkPlagiarism = document.getElementById('checkPlagiarism');
const humanize = document.getElementById('humanize');
const deplagiarize = document.getElementById('deplagiarize');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');

const fileInput = document.getElementById('fileInput');
const fileUploadArea = document.getElementById('fileUploadArea');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const removeFile = document.getElementById('removeFile');

const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

const aiResults = document.getElementById('aiResults');
const plagiarismResults = document.getElementById('plagiarismResults');
const humanizationResults = document.getElementById('humanizationResults');
const deplagResults = document.getElementById('deplagResults');

// Track uploaded file
let uploadedFile = null;

// Character counter
inputText.addEventListener('input', () => {
    const count = inputText.value.length;
    charCount.textContent = count.toLocaleString();
    if (count < 10 || count > 50000) {
        charCount.style.color = 'var(--red)';
    } else {
        charCount.style.color = 'var(--text-muted)';
    }
});

// File upload handling
fileUploadArea.addEventListener('click', (e) => {
    if (e.target !== removeFile && !removeFile.contains(e.target)) {
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        uploadedFile = file;
        fileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        fileInfo.style.display = 'flex';
        inputText.disabled = true;
        inputText.placeholder = 'File selected — text input disabled';
    }
});

removeFile.addEventListener('click', (e) => {
    e.stopPropagation();
    uploadedFile = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    inputText.disabled = false;
    inputText.placeholder = 'Paste your text here for analysis...';
});

// Drag and drop
fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadArea.classList.add('drag-over');
});
fileUploadArea.addEventListener('dragleave', () => {
    fileUploadArea.classList.remove('drag-over');
});
fileUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) {
        uploadedFile = file;
        fileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        fileInfo.style.display = 'flex';
        inputText.disabled = true;
        inputText.placeholder = 'File selected — text input disabled';
    }
});

// Clear button
clearBtn.addEventListener('click', () => {
    inputText.value = '';
    inputText.disabled = false;
    inputText.placeholder = 'Paste your text here for analysis...';
    charCount.textContent = '0';
    uploadedFile = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    hideResults();
    hideError();
});

// Copy buttons
function setupCopyBtn(btnId, outputId) {
    const btn = document.getElementById(btnId);
    if (btn) {
        btn.addEventListener('click', () => {
            const text = document.getElementById(outputId).textContent;
            navigator.clipboard.writeText(text).then(() => {
                btn.textContent = '✓ Copied!';
                setTimeout(() => { btn.textContent = 'Copy to Clipboard'; }, 2000);
            });
        });
    }
}
setupCopyBtn('copyHumanBtn', 'humanizedOutput');
setupCopyBtn('copyDeplagBtn', 'deplagOutput');

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!checkAI.checked && !checkPlagiarism.checked && !humanize.checked && !deplagiarize.checked) {
        showError('Please select at least one analysis option');
        return;
    }

    if (uploadedFile) {
        await runFileAnalysis(uploadedFile);
    } else {
        const text = inputText.value.trim();
        if (!text) { showError('Please enter some text or upload a file'); return; }
        if (text.length < 10) { showError('Text must be at least 10 characters long'); return; }
        if (text.length > 50000) { showError('Text must be less than 50,000 characters'); return; }
        await runTextAnalysis(text);
    }
});

// Run text analysis
async function runTextAnalysis(text) {
    setLoading(true);
    hideResults();
    hideError();
    try {
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                check_ai: checkAI.checked,
                check_plagiarism: checkPlagiarism.checked,
                humanize: humanize.checked,
                deplagiarize: deplagiarize.checked,
                use_cache: true
            })
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }
        displayResults(await response.json());
    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message || 'An error occurred during analysis');
    } finally {
        setLoading(false);
    }
}

// Run file analysis
async function runFileAnalysis(file) {
    setLoading(true);
    hideResults();
    hideError();
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('check_ai', checkAI.checked);
        formData.append('check_plagiarism', checkPlagiarism.checked);
        formData.append('humanize', humanize.checked);
        formData.append('deplagiarize', deplagiarize.checked);

        const response = await fetch(`${API_BASE_URL}/upload-and-analyze`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'File analysis failed');
        }
        displayResults(await response.json());
    } catch (error) {
        console.error('File analysis error:', error);
        showError(error.message || 'An error occurred during file analysis');
    } finally {
        setLoading(false);
    }
}

// Display results
function displayResults(data) {
    resultsSection.style.display = 'block';

    if (data.ai_detection) { displayAIResults(data.ai_detection); }
    else { aiResults.style.display = 'none'; }

    if (data.plagiarism) { displayPlagiarismResults(data.plagiarism); }
    else { plagiarismResults.style.display = 'none'; }

    if (data.humanization) { displayHumanizationResults(data.humanization); }
    else { humanizationResults.style.display = 'none'; }

    if (data.deplagiarization) { displayDeplagResults(data.deplagiarization); }
    else { deplagResults.style.display = 'none'; }

    document.getElementById('processingTime').textContent = data.processing_time.toFixed(2);
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayAIResults(data) {
    aiResults.style.display = 'block';
    const score = data.score * 100;
    document.getElementById('aiScoreBar').style.width = `${score}%`;
    document.getElementById('aiScore').textContent = `${score.toFixed(1)}%`;
    const labelEl = document.getElementById('aiLabel');
    labelEl.textContent = data.label === 'ai' ? 'AI-Generated' : 'Human-Written';
    labelEl.className = `label ${data.label}`;
    document.getElementById('aiConfidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
}

function displayPlagiarismResults(data) {
    plagiarismResults.style.display = 'block';
    const score = data.score * 100;
    document.getElementById('plagScoreBar').style.width = `${score}%`;
    document.getElementById('plagScore').textContent = `${score.toFixed(1)}%`;
    document.getElementById('matchCount').textContent = data.total_matches;

    const matchesList = document.getElementById('matchesList');
    matchesList.innerHTML = '';
    if (data.matches && data.matches.length > 0) {
        data.matches.forEach(match => {
            const item = document.createElement('div');
            item.className = 'match-item';
            item.innerHTML = `
                <div class="match-source">${escapeHtml(match.source)}</div>
                <div class="match-similarity">${(match.similarity * 100).toFixed(1)}% similarity</div>
            `;
            matchesList.appendChild(item);
        });
    } else {
        matchesList.innerHTML = '<p style="color: var(--text-muted);">No plagiarism detected</p>';
    }
}

function displayHumanizationResults(data) {
    humanizationResults.style.display = 'block';
    document.getElementById('humanBefore').textContent = `${(data.ai_score_before * 100).toFixed(1)}%`;
    document.getElementById('humanAfter').textContent = `${(data.ai_score_after * 100).toFixed(1)}%`;
    document.getElementById('humanIterations').textContent = data.iterations;
    document.getElementById('humanModel').textContent = data.model_used || '—';
    document.getElementById('humanizedOutput').textContent = data.text;
}

function displayDeplagResults(data) {
    deplagResults.style.display = 'block';
    document.getElementById('deplagBefore').textContent = `${(data.plagiarism_score_before * 100).toFixed(1)}%`;
    document.getElementById('deplagAfter').textContent = `${(data.plagiarism_score_after * 100).toFixed(1)}%`;
    document.getElementById('deplagRewritten').textContent = data.sentences_rewritten;
    document.getElementById('deplagIterations').textContent = data.iterations;
    document.getElementById('deplagOutput').textContent = data.text;
}

function showError(message) {
    errorSection.style.display = 'block';
    errorMessage.textContent = message;
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideError() { errorSection.style.display = 'none'; }
function hideResults() { resultsSection.style.display = 'none'; }

function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading;
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    btnText.style.display = isLoading ? 'none' : 'inline';
    btnLoader.style.display = isLoading ? 'flex' : 'none';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Check API health on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('API Health:', data);
        if (data.status !== 'healthy') {
            console.warn('API is not fully healthy:', data);
        }
    } catch (error) {
        console.error('Failed to check API health:', error);
    }
}

checkHealth();
