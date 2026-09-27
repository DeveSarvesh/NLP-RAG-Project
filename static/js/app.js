/**
 * Intelligent Multi-Document RAG Assistant - Client Application
 * Handles API communication, chat streaming, dynamic chunk rendering,
 * and passage inspection drawers.
 */

// Application State
let currentRetrievedChunks = [];
let appState = {
    documents: [],
    totalChunks: 0,
    topK: 3,
    threshold: 0.15
};

// DOM Elements
document.addEventListener("DOMContentLoaded", () => {
    initApp();
    setupEventListeners();
});

function initApp() {
    // Sync slider display
    const slider = document.getElementById("thresholdSlider");
    const badge = document.getElementById("thresholdValBadge");
    if (slider && badge) {
        badge.textContent = parseFloat(slider.value).toFixed(2);
        slider.addEventListener("input", (e) => {
            badge.textContent = parseFloat(e.target.value).toFixed(2);
            appState.threshold = parseFloat(e.target.value);
        });
    }

    const topKSelect = document.getElementById("topKSelect");
    if (topKSelect) {
        topKSelect.addEventListener("change", (e) => {
            appState.topK = parseInt(e.target.value, 10);
        });
    }

    // Fetch initial status from server
    fetchSystemStatus();
}

function setupEventListeners() {
    // Chat query form submit
    const queryForm = document.getElementById("queryForm");
    if (queryForm) {
        queryForm.addEventListener("submit", handleQuerySubmit);
    }

    // Theme Toggle
    const themeBtn = document.getElementById("themeToggle");
    if (themeBtn) {
        themeBtn.addEventListener("click", () => {
            document.body.classList.toggle("theme-dark");
        });
    }

    // File Upload Dropzone
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");

    if (dropZone && fileInput) {
        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.style.borderColor = "#2563EB";
            dropZone.style.backgroundColor = "#DBEAFE";
        });

        dropZone.addEventListener("dragleave", () => {
            dropZone.style.borderColor = "#93C5FD";
            dropZone.style.backgroundColor = "#F0F7FF";
        });

        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.style.borderColor = "#93C5FD";
            dropZone.style.backgroundColor = "#F0F7FF";
            if (e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files);
            }
        });

        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files);
            }
        });
    }
}

// Fetch System Status & Metadata
async function fetchSystemStatus() {
    try {
        const res = await fetch("/api/status");
        const data = await res.json();

        // Update Quick Stats
        document.getElementById("statDocCount").textContent = data.total_documents || 0;
        document.getElementById("statTotalChunks").textContent = data.total_chunks || 0;
        document.getElementById("statIndexSize").textContent = data.total_chunks || 0;
        document.getElementById("statModelName").textContent = data.llm_model || "gemini-2.5-flash";

        // Update Ribbon Badges
        document.getElementById("badgeChunksCount").textContent = `${data.total_chunks} chunks in index`;
        document.getElementById("badgeEmbedDim").textContent = `${data.index_dimension || 384} embedding dimension`;

        // Update Active Document Capsule & Info
        if (data.latest_document) {
            document.getElementById("capsuleDocName").textContent = data.latest_document.filename;
            document.getElementById("capsuleStatus").textContent = `${data.total_documents} document loaded`;
            
            document.getElementById("infoDocName").textContent = data.latest_document.filename;
            document.getElementById("infoTotalPages").textContent = data.latest_document.pages;
            document.getElementById("infoTotalChunks").textContent = data.latest_document.chunks;
            document.getElementById("infoProcTime").textContent = `${data.latest_document.processing_time || 2.4}s`;
        }

        appState.documents = data.documents || [];
        appState.totalChunks = data.total_chunks || 0;

        renderLibraryList();
    } catch (e) {
        console.error("Failed to fetch system status:", e);
    }
}

// Handle Natural Language Query
async function handleQuerySubmit(e) {
    e.preventDefault();
    const input = document.getElementById("questionInput");
    const question = input.value.trim();
    if (!question) return;

    // 1. Append User Message
    appendUserMessage(question);
    input.value = "";

    // 2. Append Assistant Typing Indicator
    const typingId = appendAssistantTyping();

    // 3. Send API Request
    const topK = parseInt(document.getElementById("topKSelect").value, 10);
    const threshold = parseFloat(document.getElementById("thresholdSlider").value);

    try {
        const response = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: question,
                top_k: topK,
                threshold: threshold
            })
        });

        const data = await response.json();
        removeTyping(typingId);

        if (!response.ok) {
            appendAssistantError(data.error || "Failed to process query.");
            return;
        }

        // Store retrieved chunks in memory for right panel
        currentRetrievedChunks = data.retrieval_results || [];

        // 4. Render Assistant Answer
        appendAssistantAnswer(data);

        // 5. Update Right Side Panel with Retrieved Chunks
        renderRetrievedChunks(data.retrieval_results);

        // 6. Update Processing Time
        document.getElementById("infoProcTime").textContent = `${data.execution_time_seconds}s`;

    } catch (err) {
        removeTyping(typingId);
        appendAssistantError(`Network error: ${err.message}`);
    }
}

// Render User Message
function appendUserMessage(text) {
    const container = document.getElementById("chatMessages");
    const msgDiv = document.createElement("div");
    msgDiv.className = "chat-message user-msg";
    
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    msgDiv.innerHTML = `
        <div class="msg-bubble">
            <div class="user-avatar-tag">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                </svg>
            </div>
            <p class="msg-text">${escapeHtml(text)}</p>
        </div>
        <span class="msg-timestamp">${timeStr}</span>
    `;

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

// Render Assistant Typing Indicator
function appendAssistantTyping() {
    const container = document.getElementById("chatMessages");
    const typingDiv = document.createElement("div");
    const id = "typing_" + Date.now();
    typingDiv.id = id;
    typingDiv.className = "chat-message assistant-msg";

    typingDiv.innerHTML = `
        <div class="assistant-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="11" width="18" height="10" rx="2"></rect>
                <circle cx="12" cy="5" r="2"></circle>
                <path d="M12 7v4"></path>
                <line x1="8" y1="16" x2="8" y2="16"></line>
                <line x1="16" y1="16" x2="16" y2="16"></line>
            </svg>
        </div>
        <div class="assistant-body">
            <p style="color: var(--text-muted); font-size: 13px;">Searching indexed chunks and generating grounded answer...</p>
        </div>
    `;

    container.appendChild(typingDiv);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// Render Assistant Response
function appendAssistantAnswer(data) {
    const container = document.getElementById("chatMessages");
    const msgDiv = document.createElement("div");
    msgDiv.className = "chat-message assistant-msg";

    // Format Markdown paragraphs & bullets
    const formattedAnswer = formatMarkdownToHtml(data.answer);

    // Format Sources Chips
    let sourcesHtml = "";
    if (data.sources && data.sources.length > 0) {
        const chips = data.sources.map(src => {
            return `
                <button class="source-chip" onclick="inspectCitationBySource('${escapeHtml(src)}')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                    </svg>
                    <span>${escapeHtml(src)}</span>
                </button>
            `;
        }).join("");

        sourcesHtml = `
            <div class="sources-container">
                <span class="sources-heading">Sources (${data.sources.length})</span>
                <div class="sources-chips-row">
                    ${chips}
                </div>
            </div>
        `;
    }

    const timeStr = data.timestamp || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const groundedPill = data.has_relevant_context ? `
        <div class="grounded-pill">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            <span>Grounded Answer</span>
        </div>
    ` : `
        <div class="grounded-pill" style="background: #FEF2F2; color: #991B1B; border-color: #FECACA;">
            <span>Not in Documents</span>
        </div>
    `;

    msgDiv.innerHTML = `
        <div class="assistant-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="11" width="18" height="10" rx="2"></rect>
                <circle cx="12" cy="5" r="2"></circle>
                <path d="M12 7v4"></path>
                <line x1="8" y1="16" x2="8" y2="16"></line>
                <line x1="16" y1="16" x2="16" y2="16"></line>
            </svg>
        </div>
        <div class="assistant-body">
            <div class="assistant-content">
                ${formattedAnswer}
            </div>
            <div class="assistant-footer">
                <span class="msg-timestamp">${timeStr}</span>
                ${groundedPill}
            </div>
            ${sourcesHtml}
        </div>
    `;

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function appendAssistantError(errorMsg) {
    const container = document.getElementById("chatMessages");
    const msgDiv = document.createElement("div");
    msgDiv.className = "chat-message assistant-msg";

    msgDiv.innerHTML = `
        <div class="assistant-avatar" style="background-color: #FEF2F2; color: #DC2626; border-color: #FECACA;">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
        </div>
        <div class="assistant-body" style="border-color: #FECACA; background-color: #FEF2F2;">
            <p style="color: #991B1B; font-weight: 500;">${escapeHtml(errorMsg)}</p>
        </div>
    `;

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

// Update Right Side Panel Chunks List
function renderRetrievedChunks(results) {
    const listEl = document.getElementById("retrievedChunksList");
    const countBadge = document.getElementById("rightPanelChunkCount");

    if (!results || results.length === 0) {
        listEl.innerHTML = `
            <div style="padding: 20px; text-align: center; color: var(--text-muted); font-size: 12px;">
                No relevant chunks found above threshold.
            </div>
        `;
        countBadge.textContent = "0 chunks";
        return;
    }

    countBadge.textContent = `${results.length} chunks`;

    listEl.innerHTML = results.map((item, idx) => {
        const scoreFormatted = (item.score || 0).toFixed(4);
        const snippet = escapeHtml(item.text.slice(0, 150)) + "...";

        return `
            <div class="chunk-card" onclick="inspectChunkDetail(${idx})">
                <div class="chunk-card-header">
                    <span class="chunk-meta-title">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                        </svg>
                        Page ${item.page} · Chunk ${item.rank}
                    </span>
                    <span class="similarity-score-pill">${scoreFormatted}</span>
                </div>
                <p class="chunk-text-snippet">${snippet}</p>
                <div class="chunk-card-footer">
                    <span class="expand-label">Click to view passage</span>
                    <svg class="expand-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                        <polyline points="15 3 21 3 21 9"></polyline>
                        <line x1="10" y1="14" x2="21" y2="3"></line>
                    </svg>
                </div>
            </div>
        `;
    }).join("");
}

// Inspect Chunk Details in Modal
function inspectChunkDetail(index) {
    const chunk = currentRetrievedChunks[index];
    if (!chunk) return;

    document.getElementById("modalPassageTitle").textContent = `Passage: ${chunk.document} (Page ${chunk.page})`;
    document.getElementById("modalMetaDoc").textContent = chunk.document;
    document.getElementById("modalMetaPage").textContent = `Page ${chunk.page}`;
    document.getElementById("modalMetaScore").textContent = `Similarity: ${(chunk.score || 0).toFixed(4)}`;
    document.getElementById("modalMetaWords").textContent = `${chunk.word_count || chunk.text.split(' ').length} words`;
    document.getElementById("modalPassageText").textContent = chunk.text;

    openModal("passageModal");
}

function inspectCitation(documentName, page) {
    const chunk = currentRetrievedChunks.find(c => c.document === documentName && c.page === page);
    if (chunk) {
        inspectChunkDetail(currentRetrievedChunks.indexOf(chunk));
    } else {
        document.getElementById("modalPassageTitle").textContent = `Citation: ${documentName} (Page ${page})`;
        document.getElementById("modalMetaDoc").textContent = documentName;
        document.getElementById("modalMetaPage").textContent = `Page ${page}`;
        document.getElementById("modalMetaScore").textContent = "Source Match";
        document.getElementById("modalMetaWords").textContent = "";
        document.getElementById("modalPassageText").textContent = `Direct page reference for [Doc: ${documentName}, Page: ${page}]`;
        openModal("passageModal");
    }
}

function inspectCitationBySource(sourceStr) {
    const match = currentRetrievedChunks.find(c => c.citation === sourceStr);
    if (match) {
        inspectChunkDetail(currentRetrievedChunks.indexOf(match));
    } else {
        openModal("passageModal");
    }
}

function inspectChunkByPage(pageNum) {
    const chunk = currentRetrievedChunks.find(c => c.page === pageNum);
    if (chunk) {
        inspectChunkDetail(currentRetrievedChunks.indexOf(chunk));
    } else {
        openModal("passageModal");
    }
}

// File Upload Handler
async function handleFileUpload(fileList) {
    const formData = new FormData();
    for (let i = 0; i < fileList.length; i++) {
        formData.append("files", fileList[i]);
    }

    const progressBox = document.getElementById("uploadProgress");
    const progressBar = document.getElementById("uploadProgressBar");
    const progressText = document.getElementById("uploadProgressText");

    progressBox.style.display = "block";
    progressBar.style.width = "40%";
    progressText.textContent = "Extracting text, chunking, and computing embeddings...";

    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        progressBar.style.width = "100%";

        if (!response.ok) {
            progressText.textContent = `Upload Error: ${data.error}`;
            return;
        }

        progressText.textContent = data.message;
        setTimeout(() => {
            closeModal("uploadModal");
            progressBox.style.display = "none";
            progressBar.style.width = "0%";
            fetchSystemStatus();
        }, 1200);

    } catch (e) {
        progressText.textContent = `Upload failed: ${e.message}`;
    }
}

// Reload Sample Document
async function loadSampleDoc() {
    try {
        const res = await fetch("/api/load-sample", { method: "POST" });
        const data = await res.json();
        if (data.success) {
            closeModal("libraryModal");
            fetchSystemStatus();
        }
    } catch (e) {
        console.error("Failed to load sample:", e);
    }
}

// Clear Session
async function clearSession() {
    if (!confirm("Are you sure you want to clear the entire vector index and chat history?")) return;
    try {
        const res = await fetch("/api/clear", { method: "POST" });
        const data = await res.json();
        if (data.success) {
            closeModal("settingsModal");
            document.getElementById("chatMessages").innerHTML = "";
            document.getElementById("retrievedChunksList").innerHTML = "";
            fetchSystemStatus();
        }
    } catch (e) {
        console.error("Failed to clear session:", e);
    }
}

function renderLibraryList() {
    const list = document.getElementById("libraryDocsList");
    if (!list) return;

    if (appState.documents.length === 0) {
        list.innerHTML = `<p style="color: var(--text-muted); font-size: 13px;">No documents uploaded yet.</p>`;
        return;
    }

    list.innerHTML = appState.documents.map(doc => `
        <div class="stat-card" style="margin-bottom: 8px;">
            <div class="stat-icon-wrapper">
                <svg class="stat-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
            </div>
            <div class="stat-content" style="flex: 1;">
                <span class="stat-value" style="color: var(--text-main); font-size: 13px;">${escapeHtml(doc.filename)}</span>
                <span class="stat-name">${doc.pages} pages · ${doc.chunks} chunks · ${doc.words} words</span>
            </div>
        </div>
    `).join("");
}

// Modal Helpers
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.add("active");
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.remove("active");
}

// Set Prompt from helper buttons
function setQueryPrompt(text) {
    const input = document.getElementById("questionInput");
    if (input) {
        input.value = text;
        input.focus();
    }
}

// Markdown formatting helper
function formatMarkdownToHtml(markdownText) {
    if (!markdownText) return "";

    let html = escapeHtml(markdownText);

    // Bold **text**
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Bullet points
    html = html.replace(/^\s*[-*]\s+(.*)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

    // Paragraph breaks
    html = html.replace(/\n\n+/g, '</p><p>');
    html = `<p>${html}</p>`;

    // Inline citations: [Doc: ..., Page: ...]
    html = html.replace(/\[Doc:\s*([^,]+),\s*Page:\s*(\d+)\]/g, (match, doc, page) => {
        return `<span class="citation-tag" onclick="inspectCitation('${doc.trim()}', ${page})">${match}</span>`;
    });

    return html;
}

function escapeHtml(str) {
    if (!str) return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
