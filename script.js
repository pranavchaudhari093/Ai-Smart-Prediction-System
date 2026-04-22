// Toggle Sidebar for Mobile
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    
    sidebar.classList.toggle('active');
    if (overlay) {
        overlay.classList.toggle('active');
    }
}

// Close sidebar when clicking on overlay
function closeSidebarOnOverlay() {
    const overlay = document.getElementById('sidebar-overlay');
    if (overlay) {
        overlay.addEventListener('click', () => {
            const sidebar = document.getElementById('sidebar');
            sidebar.classList.remove('active');
            overlay.classList.remove('active');
        });
    }
}

// Dynamic Greeting Logic
function updateDashboard() {
    const greetingElement = document.getElementById('greeting');
    const dateElement = document.getElementById('display-date');
    
    // Set Date
    const now = new Date();
    const dateOptions = { month: 'short', day: 'numeric', year: 'numeric' };
    dateElement.innerText = now.toLocaleDateString('en-US', dateOptions);

    // Set Greeting
    const hour = now.getHours();
    let message = "Good Evening";
    if (hour < 12) message = "Good Morning";
    else if (hour < 17) message = "Good Afternoon";
    
    // Note: The name will be injected by your backend (Flask/Django)
    // For now, we keep the text that's already there or update it
    const userName = greetingElement.innerText.replace("Welcome back,", "").trim();
    greetingElement.innerHTML = `${message}, ${userName} ✨`;
}


// Active Link Highlighter
function highlightActiveNav() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

// Initialize on Load
document.addEventListener('DOMContentLoaded', () => {
    updateDashboard();
    highlightActiveNav();
    closeSidebarOnOverlay();
    
    // Close sidebar when a nav link is clicked
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            const sidebar = document.getElementById('sidebar');
            const overlay = document.getElementById('sidebar-overlay');
            sidebar.classList.remove('active');
            if (overlay) {
                overlay.classList.remove('active');
            }
        });
    });
});