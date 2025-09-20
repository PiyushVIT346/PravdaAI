// Global variables
let uploadedFile = null;
let isLoading = false;

// DOM elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadStatus = document.getElementById('uploadStatus');
const queryInput = document.getElementById('queryInput');
const sendButton = document.getElementById('sendButton');
const chatMessages = document.getElementById('chatMessages');
const loadingOverlay = document.getElementById('loadingOverlay');
const toastContainer = document.getElementById('toastContainer');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    autoResizeTextarea();
    checkHealth();
});

// Initialize all event listeners
function initializeEventListeners() {
    // File upload events
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Query input events
    queryInput.addEventListener('keydown', handleKeyPress);
    queryInput.addEventListener('input', autoResizeTextarea);
    sendButton.addEventListener('click', sendQuery);

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.addEventListener(eventName, preventDefaults, false);
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
}

// Prevent default drag behaviors
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Handle drag over
function handleDragOver(e) {
    uploadArea.classList.add('drag-over');
}

// Handle drag leave
function handleDragLeave(e) {
    uploadArea.classList.remove('drag-over');
}

// Handle file drop
function handleDrop(e) {
    uploadArea.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle file selection
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Process selected file
function handleFile(file) {
    if (!file.type === 'application/pdf') {
        showToast('Please select a PDF file only', 'error');
        return;
    }

    if (file.size > 16 * 1024 * 1024) { // 16MB limit
        showToast('File size must be less than 16MB', 'error');
        return;
    }

    uploadFile(file);
}

// Upload file to server
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    showLoading(true);
    updateUploadStatus('Uploading and processing document...', 'info');

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            uploadedFile = result.filename;
            updateUploadStatus(`✓ Document "${result.filename}" uploaded successfully`, 'success');
            showToast('Document uploaded and processed successfully!', 'success');
        } else {
            updateUploadStatus(`✗ Upload failed: ${result.error}`, 'error');
            showToast(`Upload failed: ${result.error}`, 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        updateUploadStatus('✗ Upload failed due to network error', 'error');
        showToast('Upload failed due to network error', 'error');
    } finally {
        showLoading(false);
    }
}

// Update upload status display
function updateUploadStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = `upload-status ${type}`;
    uploadStatus.style.display = 'block';
    
    if (type === 'success') {
        setTimeout(() => {
            uploadStatus.style.display = 'none';
        }, 5000);
    }
}

// Handle enter key press in textarea
function handleKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
    }
}

// Auto-resize textarea based on content
function autoResizeTextarea() {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 150) + 'px';
    
    // Enable/disable send button based on input
    const hasText = queryInput.value.trim().length > 0;
    sendButton.disabled = !hasText || isLoading;
}

// Set query from example buttons
function setQuery(query) {
    queryInput.value = query;
    autoResizeTextarea();
    queryInput.focus();
}

// Send query to server
async function sendQuery() {
    const query = queryInput.value.trim();
    
    if (!query || isLoading) {
        return;
    }

    // Add user message to chat
    addMessage(query, 'user');
    
    // Clear input and disable button
    queryInput.value = '';
    autoResizeTextarea();
    setLoading(true);

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                uploaded_file: uploadedFile
            })
        });

        const result = await response.json();

        if (result.success) {
            // Add bot response to chat
            addMessage(result.answer, 'bot', {
                query_type: result.query_type,
                law_topic: result.law_topic
            });
        } else {
            addMessage(
                result.error || 'An error occurred while processing your query.',
                'bot',
                { error: true }
            );
            showToast('Error processing query', 'error');
        }
    } catch (error) {
        console.error('Query error:', error);
        addMessage(
            'Sorry, I encountered a network error. Please try again.',
            'bot',
            { error: true }
        );
        showToast('Network error occurred', 'error');
    } finally {
        setLoading(false);
    }
}

// Add message to chat
function addMessage(content, sender, meta = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const avatar = document.createElement('div');
    avatar.className = sender === 'user' ? 'user-avatar' : 'bot-avatar';
    avatar.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    // Format content with line breaks and basic formatting
    const formattedContent = formatMessageContent(content);
    messageContent.innerHTML = formattedContent;

    // Add metadata if available
    if (Object.keys(meta).length > 0 && !meta.error) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        
        if (meta.query_type) {
            const queryTypeSpan = document.createElement('span');
            queryTypeSpan.className = 'query-type';
            queryTypeSpan.textContent = formatQueryType(meta.query_type);
            metaDiv.appendChild(queryTypeSpan);
        }
        
        if (meta.law_topic) {
            const topicSpan = document.createElement('span');
            topicSpan.textContent = `Topic: ${formatLawTopic(meta.law_topic)}`;
            metaDiv.appendChild(topicSpan);
        }
        
        messageContent.appendChild(metaDiv);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Format message content with basic HTML
function formatMessageContent(content) {
    // Convert line breaks to HTML
    let formatted = content.replace(/\n/g, '<br>');
    
    // Format numbered lists
    formatted = formatted.replace(/^\d+\.\s(.+)$/gm, '<strong>$1</strong>');
    
    // Format bullet points
    formatted = formatted.replace(/^[\-\*]\s(.+)$/gm, '• $1');
    
    // Format bold text (basic markdown-style)
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Format italic text
    formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    return formatted;
}

// Format query type for display
function formatQueryType(queryType) {
    const types = {
        'understanding_clause_meaning': 'Clause Interpretation',
        'summary_of_document': 'Document Summary',
        'general_question_from_law': 'General Legal Q&A',
        'question_from_doc_uploaded': 'Document Analysis'
    };
    return types[queryType] || queryType;
}

// Format law topic for display
function formatLawTopic(lawTopic) {
    const topics = {
        'rental': 'Rental Law',
        'finance': 'Financial Law',
        'employment': 'Employment Law',
        'business': 'Business Law',
        'contract': 'Contract Law',
        'general': 'General Law'
    };
    return topics[lawTopic] || lawTopic;
}

// Scroll chat to bottom
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Set loading state
function setLoading(loading) {
    isLoading = loading;
    sendButton.disabled = loading || queryInput.value.trim().length === 0;
    
    if (loading) {
        sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    } else {
        sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
    }
}

// Show/hide loading overlay
function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

// Show toast notification
function showToast(message, type = 'info', duration = 5000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = getToastIcon(type);
    toast.innerHTML = `
        <i class="${icon}"></i>
        <span>${message}</span>
        <button class="toast-close" onclick="closeToast(this)">&times;</button>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto-remove toast after duration
    setTimeout(() => {
        closeToast(toast.querySelector('.toast-close'));
    }, duration);
}

// Get appropriate icon for toast type
function getToastIcon(type) {
    const icons = {
        'success': 'fas fa-check-circle',
        'error': 'fas fa-exclamation-circle',
        'info': 'fas fa-info-circle',
        'warning': 'fas fa-exclamation-triangle'
    };
    return icons[type] || icons.info;
}

// Close toast notification
function closeToast(button) {
    const toast = button.closest('.toast');
    if (toast) {
        toast.remove();
    }
}

// Clear chat messages
function clearChat() {
    // Keep only the welcome message
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    chatMessages.innerHTML = '';
    if (welcomeMessage) {
        chatMessages.appendChild(welcomeMessage);
    }
    showToast('Chat cleared', 'info');
}

// Check application health
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const result = await response.json();
        
        if (result.status === 'healthy') {
            console.log('Legal AI Assistant is healthy');
        } else {
            showToast('Service may be experiencing issues', 'warning');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        showToast('Unable to connect to service', 'error');
    }
}

// Utility function to debounce function calls
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Enhanced file validation
function validateFile(file) {
    const maxSize = 16 * 1024 * 1024; // 16MB
    const allowedTypes = ['application/pdf'];
    
    if (!allowedTypes.includes(file.type)) {
        return { valid: false, error: 'Only PDF files are supported' };
    }
    
    if (file.size > maxSize) {
        return { valid: false, error: 'File size must be less than 16MB' };
    }
    
    return { valid: true };
}

// Handle network errors gracefully
function handleNetworkError(error) {
    console.error('Network error:', error);
    
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
        return 'Unable to connect to the server. Please check your internet connection.';
    } else if (error.name === 'AbortError') {
        return 'Request was cancelled. Please try again.';
    } else {
        return 'A network error occurred. Please try again.';
    }
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K to focus on input
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        queryInput.focus();
    }
    
    // Escape to clear input
    if (e.key === 'Escape' && document.activeElement === queryInput) {
        queryInput.value = '';
        autoResizeTextarea();
        queryInput.blur();
    }
});

// Add smooth scroll behavior to chat
function smoothScrollToBottom() {
    chatMessages.scrollTo({
        top: chatMessages.scrollHeight,
        behavior: 'smooth'
    });
}

// Replace the original scrollToBottom for smooth scrolling
function scrollToBottom() {
    setTimeout(smoothScrollToBottom, 100);
}

// Handle window resize for responsive behavior
window.addEventListener('resize', debounce(function() {
    // Ensure chat scrolls to bottom on resize
    if (chatMessages.children.length > 1) {
        scrollToBottom();
    }
}, 250));

// Service worker registration for offline capabilities (if needed)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        // Uncomment the following lines if you want to add offline support
        // navigator.serviceWorker.register('/static/js/sw.js')
        //     .then(function(registration) {
        //         console.log('ServiceWorker registration successful');
        //     })
        //     .catch(function(err) {
        //         console.log('ServiceWorker registration failed: ', err);
        //     });
    });
}

// Initialize performance monitoring
const performanceObserver = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
        if (entry.entryType === 'navigation') {
            console.log(`Page load time: ${entry.loadEventEnd - entry.loadEventStart}ms`);
        }
    }
});

// Start observing performance
try {
    performanceObserver.observe({ entryTypes: ['navigation'] });
} catch (e) {
    // Performance API not supported
    console.log('Performance monitoring not available');
}