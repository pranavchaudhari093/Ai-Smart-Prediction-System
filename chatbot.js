// Global state
let isTyping = false;
let unreadCount = 0;
let isOpen = false;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initChatbot();
});

function initChatbot() {
    // Toggle button (floating)
    const chatbotToggle = document.getElementById('chatbot-toggle');
    if (chatbotToggle) {
        chatbotToggle.addEventListener('click', toggleChat);
    }
    
    // Navbar button
    const navbarChatbotBtn = document.getElementById('navbar-chatbot-btn');
    if (navbarChatbotBtn) {
        navbarChatbotBtn.addEventListener('click', toggleChat);
    }
    
    // Close button
    const closeChat = document.getElementById('close-chat');
    if (closeChat) {
        closeChat.addEventListener('click', () => {
            const container = document.getElementById('chatbot-container');
            if (container) {
                container.classList.remove('open');
            }
        });
    }
    
    // Input events
    const input = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    
    if (input && sendBtn) {
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
            if (e.key === 'Escape') {
                const container = document.getElementById('chatbot-container');
                if (container) {
                    container.classList.remove('open');
                }
            }
        });
        
        input.addEventListener('input', function() {
            sendBtn.disabled = this.value.trim() === '';
        });
        
        sendBtn.addEventListener('click', sendMessage);
    }
}

function toggleChat() {
    const container = document.getElementById('chatbot-container');
    const toggleBtn = document.getElementById('chatbot-toggle');
    const badge = toggleBtn.querySelector('.chat-badge');
    
    isOpen = !container.classList.contains('open');
    container.classList.toggle('open');
    
    if (isOpen) {
        unreadCount = 0;
        badge.style.display = 'none';
        toggleBtn.querySelector('i').style.transform = 'rotate(180deg)';
    } else {
        toggleBtn.querySelector('i').style.transform = 'rotate(0deg)';
    }
}

async function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    
    if (message === '' || isTyping) return;
    
    // Handle clear command
    if (message.toLowerCase() === 'clear') {
        clearChat();
        input.value = '';
        document.getElementById('sendBtn').disabled = true;
        return;
    }
    
    // Add user message
    addMessage(message, 'user');
    input.value = '';
    document.getElementById('sendBtn').disabled = true;
    
    // Show typing indicator with thinking message
    showTypingIndicator();
    isTyping = true;
    
    try {
        // Add thinking delay (2-3 seconds for better UX)
        const thinkingDelay = Math.random() * 1000 + 1500; // 1.5-2.5 seconds
        await new Promise(resolve => setTimeout(resolve, thinkingDelay));
        
        const response = await fetch('/chatbot_response', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) throw new Error('Network error');
        
        const data = await response.json();
        hideTypingIndicator();
        addMessage(data.reply, 'bot');
        
    } catch (error) {
        hideTypingIndicator();
        addMessage('Sorry! I encountered an error. Please try again.', 'bot');
        console.error('Chatbot error:', error);
    }
}

function addMessage(text, sender) {
    const messages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    
    messageDiv.className = sender === 'user' ? 'user-message message' : 'bot-message message';
    
    const avatarEmoji = sender === 'user' ? '👤' : '🤖';
    const senderLabel = sender === 'user' ? 'You' : 'AI Assistant';
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            ${avatarEmoji}
        </div>
        <div class="message-content">
            <strong>${senderLabel}:</strong> ${text.replace(/\n/g, '<br>')}
        </div>
    `;
    
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
    
    if (!isOpen) {
        unreadCount++;
        updateBadge();
    }
}

function showTypingIndicator() {
    const messages = document.getElementById('chat-messages');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typing-indicator';
    typingDiv.className = 'bot-message typing-indicator';
    
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-thinking">
                <span>🧠 Thinking</span>
            </div>
            <div class="typing-dots" style="margin-top: 8px;">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    messages.appendChild(typingDiv);
    messages.scrollTop = messages.scrollHeight;
}

function hideTypingIndicator() {
    isTyping = false;
    const typingDiv = document.getElementById('typing-indicator');
    if (typingDiv) {
        typingDiv.remove();
    }
}

function quickAction(message) {
    document.getElementById('userInput').value = message;
    sendMessage();
}

function clearChat() {
    const messages = document.getElementById('chat-messages');
    messages.innerHTML = `
        <div class="welcome-message">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <p><strong>AI Assistant:</strong> Chat cleared! How can I help?</p>
                <div class="quick-actions">
                    <button onclick="quickAction('What models are available?')">📊 Models</button>
                    <button onclick="quickAction('What are quick tools?')">⚡ Tools</button>
                    <button onclick="quickAction('How to generate report?')">📋 Report</button>
                    <button onclick="quickAction('Tell me about fake news')">📰 Fake News</button>
                </div>
            </div>
        </div>
    `;
}

function updateBadge() {
    const badge = document.querySelector('.chat-badge');
    badge.textContent = unreadCount;
    badge.style.display = unreadCount > 0 ? 'flex' : 'none';
}

// Auto-focus input when opened
document.addEventListener('click', function(e) {
    if (e.target.closest('#chatbot-container') && isOpen) {
        document.getElementById('userInput').focus();
    }
});
