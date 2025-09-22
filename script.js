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
    let bodyHtml = `<body><div class="container">`;
    bodyHtml += `<h1>Smart Research Assistant Report</h1>`;

    if (type === 'analysis') {
        bodyHtml += `<h2>Live Analysis of Draft</h2>`;
        bodyHtml += `<h3>Potential Citations</h3>`;
        if (data.analysis.potential_citations && data.analysis.potential_citations.length > 0) {
            bodyHtml += '<ul>';
            data.analysis.potential_citations.forEach(c => {
                bodyHtml += `<li>
                    <p><strong>Claim:</strong> "${c.claim_in_draft}"</p>
                    <blockquote>${c.supporting_quote_from_context}</blockquote>
                    <p class="source">Source: ${c.source}</p>
                </li>`;
            });
            bodyHtml += '</ul>';
        } else {
            bodyHtml += '<p>No direct citations found.</p>';
        }
        
        bodyHtml += `<h3>Unsupported Claims</h3>`;
        if (data.analysis.unsupported_claims && data.analysis.unsupported_claims.length > 0) {
             bodyHtml += '<ul>';
             data.analysis.unsupported_claims.forEach(claim => { bodyHtml += `<li>${claim}</li>`; });
             bodyHtml += '</ul>';
        } else {
            bodyHtml += '<p>All claims appear to be supported.</p>';
        }

    } else if (type === 'qa') {
        bodyHtml += `<h2>Q&A Report</h2>`;
        bodyHtml += `<p><strong>Question:</strong> ${currentReportData.question}</p>`;
        bodyHtml += `<h3>Summary</h3><p>${data.summary}</p>`;
        bodyHtml += `<h3>Sources Consulted</h3>`;
        if (data.sources && data.sources.length > 0) {
            bodyHtml += '<ul>';
            data.sources.forEach(s => {
                const sourceName = s.source || 'Unknown Source';
                const cleanSourceName = sourceName.split(/[\\/]/).pop();
                bodyHtml += `<li>
                    <p class="source"><strong>Source:</strong> ${cleanSourceName}</p>
                    <blockquote>${s.content}</blockquote>
                </li>`;
            });
            bodyHtml += '</ul>';
        } else {
            bodyHtml += "<p>No sources were consulted.</p>";
        }
    }

    bodyHtml += `</div></body>`;

    // 3. Combine everything and trigger the download as an .html file
    const fullHtml = `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Research Report</title>${styles}</head>${bodyHtml}</html>`;
    const blob = new Blob([fullHtml], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'research-report.html';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

        
// --- UI Display Functions ---

// ---  Helper to create expandable source snippets ---
function createSourceDetailHtml(sourceName, content) {
    // Handle undefined or null source names
    if (!sourceName) {
        sourceName = 'Unknown Source';
    }
    const cleanSourceName = sourceName.split(/[\\/]/).pop(); // Get just the filename
    return `
        <details class="text-sm bg-gray-900/50 rounded p-2 mt-1">
            <summary class="flex justify-between items-center font-medium text-purple-400">
                <span>${cleanSourceName}</span>
                <span class="text-xs text-gray-500">Click to expand</span>
            </summary>
            <div class="mt-2 pt-2 border-t border-gray-600 text-gray-400 whitespace-pre-wrap">${content}</div>
        </details>
    `;
}

function displayAnalysis(result) {
    const analysis = result.analysis;
    if (!analysis) return;
    analysisContent.classList.remove('hidden');
    analysisContent.innerHTML = '';

    let finalHtml = '';

    // 1. Potential Citations
    let citationsHtml = '<div class="analysis-card"><h4 class="font-semibold text-blue-400 mb-2">Potential Citations</h4>';
    if (analysis.potential_citations && analysis.potential_citations.length > 0) {
        citationsHtml += '<ul class="space-y-3 text-sm">';
        analysis.potential_citations.forEach(cit => {
            citationsHtml += `<li class="border-b border-gray-600 pb-2">
                <p class="text-gray-300"><strong>Claim:</strong> "${cit.claim_in_draft}"</p>
                <p class="text-gray-400 mt-1"><strong>Supporting Quote:</strong> "${cit.supporting_quote_from_context}"</p>
                <p class="text-xs text-purple-400 mt-1">Source: ${cit.source}</p>
            </li>`;
        });
        citationsHtml += '</ul>';
    } else {
        citationsHtml += '<p class="text-sm text-gray-500">No direct citations found for your draft.</p>';
    }
    citationsHtml += '</div>';
    finalHtml += citationsHtml;

    // 2. Unsupported Claims
    let unsupportedHtml = '<div class="analysis-card"><h4 class="font-semibold text-yellow-400 mb-2">Unsupported Claims</h4>';
    if (analysis.unsupported_claims && analysis.unsupported_claims.length > 0) {
         unsupportedHtml += '<ul class="list-disc list-inside text-sm text-gray-300 space-y-1">';
         analysis.unsupported_claims.forEach(claim => {unsupportedHtml += `<li>${claim}</li>`;});
         unsupportedHtml += '</ul><p class="text-xs text-gray-500 mt-2">These claims could not be verified.</p>';
    } else {
         unsupportedHtml += '<p class="text-sm text-gray-500">All claims appear to be supported.</p>';
    }
    unsupportedHtml += '</div>';
    finalHtml += unsupportedHtml;

    // ---  Display the raw context documents used for the analysis ---
    if (result.retrieved_context && result.retrieved_context.length > 0) {
        let contextHtml = '<div class="analysis-card"><h4 class="font-semibold text-gray-400 mb-2">Context Documents Considered</h4>';
        contextHtml += result.retrieved_context.map(s => createSourceDetailHtml(s.source, s.content)).join('');
        contextHtml += '</div>';
        finalHtml += contextHtml;
    }

    analysisContent.innerHTML = finalHtml;
}

async function handleUpload(e) {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) {
        uploadStatus.textContent = 'Please select a file.';
        uploadStatus.className = 'mt-2 text-sm text-red-400'; return;
    }

    // Disable upload button and show initial status
    uploadButton.disabled = true;
    uploadButton.textContent = 'Processing...';
    uploadStatus.textContent = 'Uploading file...';
    uploadStatus.className = 'mt-2 text-sm text-blue-400';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', { method: 'POST', body: formData });
        const result = await response.json();

        if (response.ok && result.requires_ocr) {
            // Show OCR processing message instead of progress bar
            uploadStatus.textContent = "OCR processing underway because normal parsing didn't work. Please wait...";
            uploadStatus.className = 'mt-2 text-sm text-yellow-400';
            ocrProgress.classList.add('hidden'); // Hide progress bar

            // Poll backend for OCR status
            pollOCRStatus(result.filename);

        } else {
            // Normal processing completed
            const message = response.ok ? result.success : (result.error || 'Upload failed.');
            const colorClass = response.ok ? 'text-green-400' : 'text-red-400';
            uploadStatus.textContent = message;
            uploadStatus.className = `mt-2 text-sm ${colorClass}`;
        }
    } catch (error) {
        uploadStatus.textContent = 'An error occurred during upload.';
        uploadStatus.className = 'mt-2 text-sm text-red-400';
    } finally {
        // Re-enable upload button
        uploadButton.disabled = false;
        uploadButton.textContent = 'Upload & Process';
    }
}

async function simulateOCRProgress(filename) {
    const progressSteps = [
        { progress: 10, text: 'Initializing OCR...' },
        { progress: 25, text: 'Converting PDF pages to images...' },
        { progress: 40, text: 'Processing page 1 of 3...' },
        { progress: 60, text: 'Processing page 2 of 3...' },
        { progress: 80, text: 'Processing page 3 of 3...' },
        { progress: 90, text: 'Extracting text from images...' },
        { progress: 100, text: 'Finalizing document processing...' }
    ];

    for (let step of progressSteps) {
        await new Promise(resolve => setTimeout(resolve, 1000)); // 1 second delay
        ocrProgressBar.style.width = `${step.progress}%`;
        ocrStatusText.textContent = step.text;
    }

    // Hide OCR progress and show success
    await new Promise(resolve => setTimeout(resolve, 500));
    ocrProgress.classList.add('hidden');
    uploadStatus.textContent = `File '${filename}' uploaded and processed successfully with OCR!`;
    uploadStatus.className = 'mt-2 text-sm text-green-400';
}

async function pollOCRStatus(filename) {
    const pollInterval = 2000; // 2 seconds
    return new Promise((resolve, reject) => {
        const intervalId = setInterval(async () => {
            try {
                const response = await fetch(`/api/ocr_status/${filename}`);
                const data = await response.json();
                if (data.status === "done") {
                    clearInterval(intervalId);
                    ocrProgress.classList.add('hidden');
                    uploadStatus.textContent = `File '${filename}' uploaded and processed successfully with OCR!`;
                    uploadStatus.className = 'mt-2 text-sm text-green-400';
                    resolve();
                } else if (data.status === "not_found") {
                    clearInterval(intervalId);
                    ocrProgress.classList.add('hidden');
                    uploadStatus.textContent = `OCR status not found for file '${filename}'.`;
                    uploadStatus.className = 'mt-2 text-sm text-red-400';
                    reject(new Error("OCR status not found"));
                } else {
                    // Update progress bar or status text if desired here
                    ocrStatusText.textContent = `OCR processing... Please wait.`;
                    ocrProgressBar.style.width = '50%'; // Show progress bar at 50%
                }
            } catch (error) {
                clearInterval(intervalId);
                ocrProgress.classList.add('hidden');
                uploadStatus.textContent = 'Error checking OCR status.';
                uploadStatus.className = 'mt-2 text-sm text-red-400';
                reject(error);
            }
        }, pollInterval);
    });
}

async function updateStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        questionsCounter.textContent = stats.questions_asked;
        reportsCounter.textContent = stats.reports_generated;

        // Update stats in learning section if it exists
        const questionsCounter2 = document.getElementById('questions-counter-2');
        const reportsCounter2 = document.getElementById('reports-counter-2');
        if (questionsCounter2) questionsCounter2.textContent = stats.questions_asked;
        if (reportsCounter2) reportsCounter2.textContent = stats.reports_generated;
    } catch (error) {
        console.error('Failed to fetch stats:', error);
    }
}

// --- AUTHENTICATION FUNCTIONS ---
let currentUser = null;

async function checkAuthStatus() {
    try {
        const response = await fetch('/api/auth/status');
        const data = await response.json();
        currentUser = data.logged_in ? { username: data.username } : null;
        updateAuthUI();
    } catch (error) {
        console.error('Auth status check failed:', error);
        currentUser = null;
        updateAuthUI();
    }
}

function updateAuthUI() {
    const authSection = document.getElementById('auth-section');
    const learningSection = document.getElementById('learning-section');
    if (currentUser) {
        authSection.innerHTML = `
            <div class="flex items-center space-x-4">
                <span class="text-green-400">Welcome, ${currentUser.username}!</span>
                <button id="logout-btn" class="bg-red-600 hover:bg-red-700 text-white font-bold py-1 px-3 text-sm rounded-lg transition duration-300">
                    Logout
                </button>
            </div>
        `;
        document.getElementById('logout-btn').addEventListener('click', handleLogout);

        // Show learning section for authenticated users
        learningSection.classList.remove('hidden');
    } else {
        authSection.innerHTML = `
            <div class="flex items-center space-x-2">
                <button id="login-btn" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-1 px-3 text-sm rounded-lg transition duration-300">
                    Login
                </button>
                <button id="register-btn" class="bg-purple-600 hover:bg-purple-700 text-white font-bold py-1 px-3 text-sm rounded-lg transition duration-300">
                    Register
                </button>
            </div>
        `;
        document.getElementById('login-btn').addEventListener('click', showLoginModal);
        document.getElementById('register-btn').addEventListener('click', showRegisterModal);

        // Hide learning section for non-authenticated users
        learningSection.classList.add('hidden');
    }
}

function showLoginModal() {
    const modal = createAuthModal('Login', `
        <form id="login-form">
            <div class="mb-4">
                <label class="block text-gray-300 text-sm font-bold mb-2">Username</label>
                <input type="text" id="login-username" class="w-full bg-gray-700 text-white p-3 rounded-lg border border-gray-600 focus:ring-2 focus:ring-blue-500 focus:outline-none" placeholder="Enter your username">
            </div>
            <div class="mb-6">
                <label class="block text-gray-300 text-sm font-bold mb-2">Password</label>
                <input type="password" id="login-password" class="w-full bg-gray-700 text-white p-3 rounded-lg border border-gray-600 focus:ring-2 focus:ring-blue-500 focus:outline-none" placeholder="Enter your password">
            </div>
            <div class="flex items-center justify-between">
                <button type="submit" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition duration-300">
                    Login
                </button>
                <button type="button" id="switch-to-register" class="text-blue-400 hover:text-blue-300 text-sm">
                    Need an account?
                </button>
            </div>
        </form>
    `);
    document.body.appendChild(modal);

    document.getElementById('login-form').addEventListener('submit', handleLogin);
    document.getElementById('switch-to-register').addEventListener('click', () => {
        modal.remove();
        showRegisterModal();
    });
}

function showRegisterModal() {
    const modal = createAuthModal('Register', `
        <form id="register-form">
            <div class="mb-4">
                <label class="block text-gray-300 text-sm font-bold mb-2">Username</label>
                <input type="text" id="register-username" class="w-full bg-gray-700 text-white p-3 rounded-lg border border-gray-600 focus:ring-2 focus:ring-blue-500 focus:outline-none" placeholder="Choose a username">
            </div>
            <div class="mb-4">
                <label class="block text-gray-300 text-sm font-bold mb-2">Password</label>
                <input type="password" id="register-password" class="w-full bg-gray-700 text-white p-3 rounded-lg border border-gray-600 focus:ring-2 focus:ring-blue-500 focus:outline-none" placeholder="Choose a password">
            </div>
            <div class="mb-6">
                <label class="block text-gray-300 text-sm font-bold mb-2">Confirm Password</label>
                <input type="password" id="register-confirm-password" class="w-full bg-gray-700 text-white p-3 rounded-lg border border-gray-600 focus:ring-2 focus:ring-blue-500 focus:outline-none" placeholder="Confirm your password">
            </div>
            <div class="flex items-center justify-between">
                <button type="submit" class="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded-lg transition duration-300">
                    Register
                </button>
                <button type="button" id="switch-to-login" class="text-purple-400 hover:text-purple-300 text-sm">
                    Already have an account?
                </button>
            </div>
        </form>
    `);
    document.body.appendChild(modal);

    document.getElementById('register-form').addEventListener('submit', handleRegister);
    document.getElementById('switch-to-login').addEventListener('click', () => {
        modal.remove();
        showLoginModal();
    });
}

function createAuthModal(title, content) {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="bg-gray-800 p-8 rounded-2xl shadow-xl max-w-md w-full mx-4">
            <h2 class="text-2xl font-bold text-white mb-6">${title}</h2>
            ${content}
            <button id="close-modal" class="absolute top-4 right-4 text-gray-400 hover:text-white text-2xl">&times;</button>
        </div>
    `;

    // Add event listener to close button after modal is added to DOM
    const closeButton = modal.querySelector('#close-modal');
    if (closeButton) {
        closeButton.addEventListener('click', () => modal.remove());
    }

    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
    return modal;
}

async function handleLogin(e) {
    e.preventDefault();
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;

    if (!username || !password) {
        alert('Please fill in all fields');
        return;
    }

    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        const result = await response.json();

        if (result.success) {
            currentUser = { username: result.username };
            updateAuthUI();
            document.querySelector('.fixed.inset-0').remove();
            updateStats(); // Refresh stats after login
        } else {
            alert(result.error || 'Login failed');
        }
    } catch (error) {
        alert('Login failed. Please try again.');
    }
}

async function handleRegister(e) {
    e.preventDefault();
    const username = document.getElementById('register-username').value;
    const password = document.getElementById('register-password').value;
    const confirmPassword = document.getElementById('register-confirm-password').value;

    if (!username || !password || !confirmPassword) {
        alert('Please fill in all fields');
        return;
    }

    if (password !== confirmPassword) {
        alert('Passwords do not match');
        return;
    }

    try {
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        const result = await response.json();

        if (result.success) {
            alert('Registration successful! You can now log in.');
            document.querySelector('.fixed.inset-0').remove();
            showLoginModal();
        } else {
            alert(result.error || 'Registration failed');
        }
    } catch (error) {
        alert('Registration failed. Please try again.');
    }
}

async function handleLogout() {
    try {
        await fetch('/api/auth/logout', { method: 'POST' });
        currentUser = null;
        updateAuthUI();
        updateStats(); // Refresh stats after logout
    } catch (error) {
        console.error('Logout failed:', error);
    }
}

// --- LEARNING MODULE FUNCTIONS ---
async function searchConcept() {
    const topic = prompt('Enter a topic to search for:');
    if (!topic) return;

    try {
        const response = await fetch('/api/learn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic })
        });
        const result = await response.json();

        if (result.success) {
            displayConceptResults(topic, result.papers);
        } else {
            alert(result.error || 'Search failed');
        }
    } catch (error) {
        alert('Search failed. Please try again.');
    }
}

function displayConceptResults(topic, papers) {
    const resultsDiv = document.createElement('div');
    resultsDiv.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
    resultsDiv.innerHTML = `
        <div class="bg-gray-800 p-8 rounded-2xl shadow-xl max-w-2xl w-full mx-4 max-h-96 overflow-y-auto">
            <h3 class="text-xl font-bold text-white mb-4">Results for: ${topic}</h3>
            <div class="space-y-2">
                ${papers.length > 0 ?
                    papers.map(paper => `
                        <div class="bg-gray-700 p-3 rounded-lg">
                            <p class="text-gray-300">${paper.path.split('/').pop()}</p>
                            <button onclick="summarizePaper('${paper.path}')" class="text-blue-400 hover:text-blue-300 text-sm mt-1">
                                Summarize
                            </button>
                        </div>
                    `).join('') :
                    '<p class="text-gray-400">No papers found for this topic.</p>'
                }
            </div>
            <button onclick="this.parentElement.parentElement.remove()" class="mt-4 bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg">
                Close
            </button>
        </div>
    `;
    document.body.appendChild(resultsDiv);
}

async function summarizePaper(filepath) {
    try {
        const response = await fetch('/api/summarize_paper', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filepath })
        });
        const result = await response.json();

        if (result.success) {
            alert(`Summary: ${result.summary}`);
        } else {
            alert(result.error || 'Summarization failed');
        }
    } catch (error) {
        alert('Summarization failed. Please try again.');
    }
}

// --- LIBRARY FUNCTIONS ---
async function showLibrary() {
    try {
        const response = await fetch('/api/library');
        const papers = await response.json();

        const libraryDiv = document.createElement('div');
        libraryDiv.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        libraryDiv.innerHTML = `
            <div class="bg-gray-800 p-8 rounded-2xl shadow-xl max-w-2xl w-full mx-4 max-h-96 overflow-y-auto">
                <h3 class="text-xl font-bold text-white mb-4">My Library</h3>
                <div class="space-y-2">
                    ${papers.length > 0 ?
                        papers.map(paper => `
                            <div class="bg-gray-700 p-3 rounded-lg flex justify-between items-center">
                                <span class="text-gray-300">${paper.file_name}</span>
                                <div class="space-x-2">
                                    <button onclick="summarizePaper('${paper.file_path}')" class="text-blue-400 hover:text-blue-300 text-sm">
                                        Summarize
                                    </button>
                                    <a href="/download/${encodeURIComponent(paper.file_name)}" class="text-green-400 hover:text-green-300 text-sm">
                                        Download
                                    </a>
                                </div>
                            </div>
                        `).join('') :
                        '<p class="text-gray-400">Your library is empty.</p>'
                    }
                </div>
                <button onclick="this.parentElement.parentElement.remove()" class="mt-4 bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg">
                    Close
                </button>
            </div>
        `;
        document.body.appendChild(libraryDiv);
    } catch (error) {
        alert('Failed to load library. Please try again.');
    }
}

// --- Event Listeners ---
draftEditor.addEventListener('input', debouncedAnalyze);
uploadForm.addEventListener('submit', handleUpload);
questionForm.addEventListener('submit', askQuestion);
exportButton.addEventListener('click', exportReport);

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    updateStats();
    checkAuthStatus().catch(() => {
        // If auth check fails, still set up the UI for non-authenticated users
        console.log('Auth check failed, setting up UI for guest user');
        updateAuthUI();
    });
});

// --- DOCUMENT ORGANIZATION FUNCTIONS ---

// Show/hide document organization section
function showDocumentOrganization() {
    const orgSection = document.getElementById('document-organization-section');
    if (orgSection.classList.contains('hidden')) {
        orgSection.classList.remove('hidden');
        loadDocuments();
        loadCategoriesAndTags();
    } else {
        orgSection.classList.add('hidden');
    }
}

// Load documents with attribution information
async function loadDocuments() {
    const documentsList = document.getElementById('documents-list');
    const loadingIndicator = document.getElementById('documents-loading');
    const noDocuments = document.getElementById('no-documents');

    // Show loading indicator
    documentsList.innerHTML = '';
    loadingIndicator.classList.remove('hidden');
    noDocuments.classList.add('hidden');

    try {
        const response = await fetch('/api/documents');
        const documents = await response.json();

        loadingIndicator.classList.add('hidden');

        if (documents.length === 0) {
            noDocuments.classList.remove('hidden');
            return;
        }

        displayDocuments(documents);
    } catch (error) {
        console.error('Failed to load documents:', error);
        loadingIndicator.classList.add('hidden');
        documentsList.innerHTML = '<p class="text-red-400">Failed to load documents. Please try again.</p>';
    }
}

// Display documents with attribution information
function displayDocuments(documents) {
    const documentsList = document.getElementById('documents-list');
    documentsList.innerHTML = '';

    documents.forEach(doc => {
        const docCard = document.createElement('div');
        docCard.className = 'bg-gray-700 p-4 rounded-lg border border-gray-600';

        // Format upload date
        const uploadDate = new Date(doc.uploaded_at).toLocaleDateString();

        // Get category and tag names
        const categoryNames = doc.categories ? doc.categories.map(c => c.name).join(', ') : 'None';
        const tagNames = doc.tags ? doc.tags.map(t => t.name).join(', ') : 'None';

        docCard.innerHTML = `
            <div class="flex justify-between items-start mb-3">
                <div class="flex-1">
                    <h4 class="font-semibold text-white mb-1">${doc.title || doc.filename}</h4>
                    <p class="text-sm text-gray-400 mb-2">${doc.description || 'No description'}</p>
                </div>
                <div class="flex space-x-2">
                    <button onclick="showDocumentDetails(${doc.id})" class="text-blue-400 hover:text-blue-300 text-sm">
                        Details
                    </button>
                    <button onclick="downloadDocument('${doc.filename}')" class="text-green-400 hover:text-green-300 text-sm">
                        Download
                    </button>
                </div>
            </div>

            <div class="grid grid-cols-2 gap-4 text-sm">
                <div>
                    <span class="text-gray-400">Uploaded by:</span>
                    <span class="text-purple-400 ml-1">${doc.uploaded_by_username}</span>
                </div>
                <div>
                    <span class="text-gray-400">Upload Date:</span>
                    <span class="text-gray-300 ml-1">${uploadDate}</span>
                </div>
                <div>
                    <span class="text-gray-400">Categories:</span>
                    <span class="text-blue-400 ml-1">${categoryNames}</span>
                </div>
                <div>
                    <span class="text-gray-400">Tags:</span>
                    <span class="text-green-400 ml-1">${tagNames}</span>
                </div>
            </div>

            <div class="mt-3 flex justify-between items-center">
                <div class="text-xs text-gray-500">
                    ${doc.file_size} bytes • ${doc.file_type}
                </div>
                <div class="flex space-x-2">
                    <button onclick="addToCategory(${doc.id})" class="text-blue-400 hover:text-blue-300 text-xs">
                        + Category
                    </button>
                    <button onclick="addTag(${doc.id})" class="text-green-400 hover:text-green-300 text-xs">
                        + Tag
                    </button>
                </div>
            </div>
        `;

        documentsList.appendChild(docCard);
    });
}

// Load categories and tags for filter dropdowns
async function loadCategoriesAndTags() {
    try {
        // Load categories
        const catResponse = await fetch('/api/categories');
        const categories = await catResponse.json();

        // Load tags
        const tagResponse = await fetch('/api/tags');
        const tags = await tagResponse.json();

        // Populate category filter dropdown
        const categoryFilter = document.getElementById('category-filter');
        categoryFilter.innerHTML = '<option value="">All Categories</option>' +
            categories.map(cat => `<option value="${cat.id}">${cat.name}</option>`).join('');

        // Populate tag filter dropdown
        const tagFilter = document.getElementById('tag-filter');
        tagFilter.innerHTML = '<option value="">All Tags</option>' +
            tags.map(tag => `<option value="${tag.id}">${tag.name}</option>`).join('');

    } catch (error) {
        console.error('Failed to load categories and tags:', error);
    }
}

// Search documents with filters
async function searchDocuments() {
    const searchTerm = document.getElementById('doc-search-input').value;
    const categoryId = document.getElementById('category-filter').value;
    const tagId = document.getElementById('tag-filter').value;

    const documentsList = document.getElementById('documents-list');
    const loadingIndicator = document.getElementById('documents-loading');
    const noDocuments = document.getElementById('no-documents');

    // Show loading indicator
    documentsList.innerHTML = '';
    loadingIndicator.classList.remove('hidden');
    noDocuments.classList.add('hidden');

    try {
        let url = '/api/documents';
        const params = new URLSearchParams();
        if (searchTerm) params.append('search', searchTerm);
        if (categoryId) params.append('category', categoryId);
        if (tagId) params.append('tag', tagId);

        if (params.toString()) {
            url += '?' + params.toString();
        }

        const response = await fetch(url);
        const documents = await response.json();

        loadingIndicator.classList.add('hidden');

        if (documents.length === 0) {
            noDocuments.classList.remove('hidden');
            return;
        }

        displayDocuments(documents);
    } catch (error) {
        console.error('Failed to search documents:', error);
        loadingIndicator.classList.add('hidden');
        documentsList.innerHTML = '<p class="text-red-400">Failed to search documents. Please try again.</p>';
    }
}

// Create modal helper function
function createModal(title, content) {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="bg-gray-800 p-8 rounded-2xl shadow-xl max-w-md w-full mx-4">
            <h2 class="text-2xl font-bold text-white mb-6">${title}</h2>
            ${content}
        </div>
    `;

    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });
    return modal;
}

// Close modal helper function
function closeModal() {
    const modal = document.querySelector('.fixed.inset-0.bg-black.bg-opacity-50');
    if (modal) modal.remove();
}

// Show document details
function showDocumentDetails(documentId) {
    // This would show detailed information about a document
    // For now, just show a simple alert
    alert(`Document details for ID: ${documentId}\n(This feature can be expanded to show more detailed information)`);
}

// Download document
function downloadDocument(filename) {
    window.open(`/download/${encodeURIComponent(filename)}`, '_blank');
}

// Add to category modal
function addToCategory(documentId) {
    const modal = createModal('Add to Category', `
        <form id="add-to-category-form">
            <div class="mb-4">
                <label class="block text-gray-300 text-sm font-bold mb-2">Select Category</label>
                <select id="category-select" class="w-full bg-gray-700 text-white p-3 rounded-lg border border-gray-600 focus:ring-2 focus:ring-blue-500 focus:outline-none" required>
                    <option value="">Choose a category...</option>
                </select>
            </div>
            <div class="flex items-center justify-between">
                <button type="submit" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition duration-300">
                    Add to Category
                </button>
                <button type="button" onclick="closeModal()" class="text-gray-400 hover:text-white">
                    Cancel
                </button>
            </div>
        </form>
    `);
    document.body.appendChild(modal);

    // Load categories into the select
    loadCategoriesIntoSelect();

    document.getElementById('add-to-category-form').addEventListener('submit', (e) => handleAddToCategory(e, documentId));
}

// Load categories into select dropdown
async function loadCategoriesIntoSelect() {
    try {
        const response = await fetch('/api/categories');
        const categories = await response.json();

        const select = document.getElementById('category-select');
        select.innerHTML = '<option value="">Choose a category...</option>' +
            categories.map(cat => `<option value="${cat.id}">${cat.name}</option>`).join('');
    } catch (error) {
        console.error('Failed to load categories:', error);
    }
}

// Handle adding document to category
async function handleAddToCategory(e, documentId) {
    e.preventDefault();
    const categoryId = document.getElementById('category-select').value;

    if (!categoryId) {
        alert('Please select a category');
        return;
    }

    try {
        const response = await fetch(`/api/documents/${documentId}/categories`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category_id: parseInt(categoryId) })
        });
        const result = await response.json();

        if (result.success) {
            closeModal();
            await loadDocuments(); // Refresh documents to show new category assignment
            alert('Document added to category successfully!');
        } else {
            alert(result.error || 'Failed to add document to category');
        }
    } catch (error) {
        alert('Failed to add document to category. Please try again.');
    }
}

// Add tag modal
function addTag(documentId) {
    const modal = createModal('Add Tag', `
        <form id="add-tag-form">
            <div class="mb-4">
                <label class="block text-gray-300 text-sm font-bold mb-2">Select Tag</label>
                <select id="tag-select" class="w-full bg-gray-700 text-white p-3 rounded-lg border border-gray-600 focus:ring-2 focus:ring-blue-500 focus:outline-none" required>
                    <option value="">Choose a tag...</option>
                </select>
            </div>
            <div class="flex items-center justify-between">
                <button type="submit" class="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg transition duration-300">
                    Add Tag
                </button>
                <button type="button" onclick="closeModal()" class="text-gray-400 hover:text-white">
                    Cancel
                </button>
            </div>
        </form>
    `);
    document.body.appendChild(modal);

    // Load tags into the select
    loadTagsIntoSelect();

    document.getElementById('add-tag-form').addEventListener('submit', (e) => handleAddTag(e, documentId));
}

// Load tags into select dropdown
async function loadTagsIntoSelect() {
    try {
        const response = await fetch('/api/tags');
        const tags = await response.json();

        const select = document.getElementById('tag-select');
        select.innerHTML = '<option value="">Choose a tag...</option>' +
            tags.map(tag => `<option value="${tag.id}">${tag.name}</option>`).join('');
    } catch (error) {
        console.error('Failed to load tags:', error);
    }
}

// Handle adding tag to document
async function handleAddTag(e, documentId) {
    e.preventDefault();
    const tagId = document.getElementById('tag-select').value;

    if (!tagId) {
        alert('Please select a tag');
        return;
    }

    try {
        const response = await fetch(`/api/documents/${documentId}/tags`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tag_id: parseInt(tagId) })
        });
        const result = await response.json();

        if (result.success) {
            closeModal();
            await loadDocuments(); // Refresh documents to show new tag assignment
            alert('Tag added to document successfully!');
        } else {
            alert(result.error || 'Failed to add tag to document');
        }
    } catch (error) {
        alert('Failed to add tag to document. Please try again.');
    }
}

// Make functions globally available for onclick handlers
window.searchConcept = searchConcept;
window.summarizePaper = summarizePaper;
window.showLibrary = showLibrary;
window.showDocumentOrganization = showDocumentOrganization;
window.showCreateCategoryModal = showCreateCategoryModal;
window.showCreateTagModal = showCreateTagModal;
window.searchDocuments = searchDocuments;
window.closeModal = closeModal;
window.showDocumentDetails = showDocumentDetails;
window.downloadDocument = downloadDocument;
window.addToCategory = addToCategory;
window.addTag = addTag;
