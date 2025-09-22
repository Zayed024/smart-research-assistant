// --- DOM Elements ---
const uploadForm = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const uploadButton = document.getElementById('upload-button');
const uploadStatus = document.getElementById('upload-status');
const ocrProgress = document.getElementById('ocr-progress');
const ocrProgressBar = document.getElementById('ocr-progress-bar');
const ocrStatusText = document.getElementById('ocr-status-text');
const questionsCounter = document.getElementById('questions-counter');
const reportsCounter = document.getElementById('reports-counter');
const draftEditor = document.getElementById('draft-editor');
const questionForm = document.getElementById('question-form');
const questionInput = document.getElementById('question-input');
const analysisContent = document.getElementById('analysis-content');
const analysisPlaceholder = document.getElementById('analysis-placeholder');
const analysisLoader = document.getElementById('analysis-loader');
const qaResponse = document.getElementById('qa-response');
const exportButton = document.getElementById('export-button');

let currentReportData = null;
// --- Debounce Utility ---
function debounce(func, delay) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), delay);
    };
}

// --- Core Functions ---

async function analyzeDraft() {
    const draftContent = draftEditor.value.trim();
    if (draftContent.length < 50) {
        analysisPlaceholder.classList.remove('hidden');
        analysisContent.classList.add('hidden');
        return;
    }
    analysisPlaceholder.classList.add('hidden');
    analysisLoader.classList.remove('hidden');
    analysisContent.classList.add('hidden');
    try {
        const response = await fetch('/api/live_analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ draft_content: draftContent }),
        });
        const result = await response.json();
        //if (response.ok) {
            //displayAnalysis(result); // Pass the whole result object
        //}
        if (response.ok && result.success) {
            currentReportData = { type: 'analysis', data: result,draft: draftEditor.value.trim() }; // Store data for export
            exportButton.classList.remove('hidden'); // Show export button
            displayAnalysis(result);
        }
    } catch (error) {
        console.error('Analysis Error:', error);
    } finally {
        analysisLoader.classList.add('hidden');
    }
}
const debouncedAnalyze = debounce(analyzeDraft, 1500);

async function askQuestion(e) {
    e.preventDefault();
    const question = questionInput.value.trim();
    const draftContext = draftEditor.value.trim();
    if (!question) return;
    qaResponse.innerHTML = `<p class="text-sm text-blue-400">Getting answer...</p>`;
    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, draft_context: draftContext }),
        });
        const result = await response.json();
        // --- UPDATED: Render Q&A response with sources ---
        if (result.success) {
            currentReportData = { type: 'qa', data: result, question: questionInput.value.trim() }; // Store data for export
            exportButton.classList.remove('hidden'); // Show export button

            let sourcesHtml = '';
            if (result.sources && result.sources.length > 0) {
                sourcesHtml = '<h4 class="font-semibold text-gray-400 mt-4 mb-2">Sources Consulted</h4>' +
                    result.sources.map(s => createSourceDetailHtml(s.source, s.content)).join('');
            }
            qaResponse.innerHTML = `<div class="analysis-card">
                <p class="text-gray-300">${result.summary}</p>
                ${sourcesHtml}
            </div>`;
        } else {
            // Handle specific error cases with user-friendly messages
            let errorMessage = result.error || 'Failed to get answer.';
            if (errorMessage.includes('Please upload a source document first')) {
                errorMessage = 'Please upload a source document first before asking questions.';
            }
            qaResponse.innerHTML = `<p class="text-sm text-red-400">${errorMessage}</p>`;
        }
    } catch (error) {
        console.error('Q&A Error:', error);
        let errorMessage = 'An error occurred. Please try again.';
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            errorMessage = 'Network error. Please check your connection and try again.';
        } else if (error.name === 'SyntaxError') {
            errorMessage = 'Invalid response from server. Please try again.';
        }
        qaResponse.innerHTML = `<p class="text-sm text-red-400">${errorMessage}</p>`;
    } finally {
        updateStats();
    }
}


function exportReport() {
    if (!currentReportData) {
        alert("No report data available to export.");
        return;
    }


    const { type, data } = currentReportData;

    const styles = `
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #111827; color: #d1d5db; line-height: 1.6; }
            .container { max-width: 800px; margin: 2rem auto; padding: 2rem; background-color: #1f2937; border-radius: 8px; border: 1px solid #374151; }
            h1 { color: #93c5fd; border-bottom: 2px solid #3b82f6; padding-bottom: 0.5rem; }
            h2 { color: #a5b4fc; border-bottom: 1px solid #4f46e5; padding-bottom: 0.3rem; margin-top: 2rem; }
            h3 { color: #f9a8d4; }
            ul { list-style-type: none; padding-left: 0; }
            li { background-color: #374151; padding: 1rem; border-radius: 6px; margin-bottom: 1rem; }
            blockquote { border-left: 4px solid #6b7280; padding-left: 1rem; margin-left: 0; font-style: italic; color: #9ca3af; }
            .source { font-size: 0.8rem; color: #a5b4fc; }
        </style>
    `;

    // 2. Build the HTML body of the report
