// Initialize Bootstrap tooltips
document.addEventListener('DOMContentLoaded', function() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// script.js
function playAudio() {
    var audio = document.getElementById('audio-player');
    audio.style.display = 'block'; // Show the audio player
    audio.play();
}

// Function to dynamically load Chart.js if needed
function loadChartJS() {
    return new Promise((resolve, reject) => {
        if (typeof Chart !== 'undefined') {
            resolve(); // Chart.js is already loaded
        } else {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
            script.onload = () => resolve();
            script.onerror = () => reject('Failed to load Chart.js');
            document.head.appendChild(script);
        }
    });
}

// Initialize the chart
async function initializeChart() {
    await loadChartJS();

    // Check if the chart canvas exists
    const chartCanvas = document.getElementById('trialsChart');
    if (chartCanvas) {
        const chartData = JSON.parse(chartCanvas.getAttribute('data-chart'));
        const ctx = chartCanvas.getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: chartData.labels,
                datasets: [{
                    label: 'Number of Trials',
                    data: chartData.values,
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

// Initialize tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map((tooltipTriggerEl) => {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Initialize the dashboard
function initializeDashboard() {
    initializeTooltips();
    initializeChart();
}

// Function to render chart
async function renderChart() {
    await loadChartJS();

    const chartCanvas = document.getElementById('trialsChart');
    if (!chartCanvas) return;

    const ctx = chartCanvas.getContext('2d');

    const dataElement = document.getElementById('chartData');
    if (!dataElement) return;

    let chartData;
    try {
        chartData = JSON.parse(dataElement.getAttribute('data-chart'));
    } catch (e) {
        console.error('Failed to parse chart data:', e);
        return;
    }

    if (!chartData.labels || !chartData.values) {
        console.error('Invalid chart data structure.');
        return;
    }

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: 'Number of Trials Over Time',
                data: chartData.values,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: {
                        autoSkip: false
                    }
                },
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(tooltipItem) {
                            return `Trials: ${tooltipItem.raw}`;
                        }
                    }
                },
                legend: {
                    display: true
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuad'
            }
        }
    });
}

// Call the renderChart function on page load
document.addEventListener('DOMContentLoaded', function() {
    renderChart().catch(error => console.error(error));
});
