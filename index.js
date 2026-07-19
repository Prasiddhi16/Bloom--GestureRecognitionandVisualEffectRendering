// -------------------- Dummy Data --------------------
const busNetwork = {
    stops: {
        'Stop A': { x: 100, y: 150, color: '#4CAF50' },
        'Stop B': { x: 200, y: 200, color: '#4CAF50' },
        'Stop C': { x: 300, y: 100, color: '#FFC107' },
        'Stop D': { x: 300, y: 300, color: '#FF6B6B' },
        'Stop E': { x: 150, y: 280, color: '#4CAF50' },
        'Stop F': { x: 450, y: 150, color: '#4CAF50' },
        'Stop G': { x: 550, y: 250, color: '#4CAF50' },
        'Stop H': { x: 650, y: 100, color: '#4CAF50' },
    },
    routes: [
        { from: 'Stop A', to: 'Stop B', distance: 1.2, time: 4 },
        { from: 'Stop B', to: 'Stop C', distance: 1.3, time: 5 },
        { from: 'Stop A', to: 'Stop E', distance: 1.5, time: 5 },
        { from: 'Stop E', to: 'Stop D', distance: 0.8, time: 3 },
        { from: 'Stop C', to: 'Stop F', distance: 1.8, time: 6 },
        { from: 'Stop F', to: 'Stop G', distance: 1.2, time: 4 },
        { from: 'Stop G', to: 'Stop H', distance: 1.5, time: 5 },
        { from: 'Stop D', to: 'Stop G', distance: 2.5, time: 8 },
    ]
};

const routeResults = {
    distance: '5 km',
    time: '14 min',
    cost: '$18',
    transfers: 0,
    stops: ['Stop A', 'Stop E', 'Stop D', 'Stop G'],
    segments: [
        {
            from: 'Stop A',
            to: 'Stop D',
            via: ['Stop A', 'Stop B', 'Stop C', 'Stop D'],
            busNumber: '101',
            distance: '2.5 km',
            time: '8 min'
        },
        {
            from: 'Stop D',
            to: 'Stop G',
            via: ['Stop D', 'Stop G'],
            busNumber: '102',
            distance: '2.5 km',
            time: '6 min'
        }
    ]
};

// -------------------- DOM Elements --------------------
const findRouteBtn = document.querySelector('.btn-primary');
const clearBtn = document.querySelector('.btn-secondary');
const busNetworkCanvas = document.getElementById('busNetworkCanvas');
const routeSegmentsDiv = document.querySelector('.route-segments');
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const musicCards = document.querySelectorAll('.music-card');
const optionButtons = document.querySelectorAll('.option-btn');

// -------------------- Draw Bus Network --------------------
function drawBusNetwork() {
    const canvas = busNetworkCanvas;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Background
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, width, height);

    // Grid
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < width; i += 50) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, height);
        ctx.stroke();
    }
    for (let i = 0; i < height; i += 50) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(width, i);
        ctx.stroke();
    }

    // Routes
    ctx.strokeStyle = '#DDD';
    ctx.lineWidth = 2;
    busNetwork.routes.forEach(route => {
        const from = busNetwork.stops[route.from];
        const to = busNetwork.stops[route.to];
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();

        const midX = (from.x + to.x) / 2;
        const midY = (from.y + to.y) / 2;
        ctx.fillStyle = '#999';
        ctx.font = '11px Arial';
        ctx.fillText(route.distance + 'km', midX - 15, midY - 5);
    });

    // Optimal Route Highlight
    const optimalRoute = routeResults.stops.map(stop => busNetwork.stops[stop]);
    ctx.strokeStyle = '#FF6B6B';
    ctx.lineWidth = 3;
    ctx.setLineDash([5, 5]);
    for (let i = 0; i < optimalRoute.length - 1; i++) {
        ctx.beginPath();
        ctx.moveTo(optimalRoute[i].x, optimalRoute[i].y);
        ctx.lineTo(optimalRoute[i + 1].x, optimalRoute[i + 1].y);
        ctx.stroke();
    }
    ctx.setLineDash([]);

    // Draw Stops
    Object.keys(busNetwork.stops).forEach(stopName => {
        const stop = busNetwork.stops[stopName];
        ctx.fillStyle = '#4CAF50';
        ctx.beginPath();
        ctx.arc(stop.x, stop.y, 8, 0, 2 * Math.PI);
        ctx.fill();

        if (routeResults.stops.includes(stopName)) {
            ctx.fillStyle = '#FF6B6B';
            ctx.beginPath();
            ctx.arc(stop.x, stop.y, 10, 0, 2 * Math.PI);
            ctx.fill();
        }

        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(stopName, stop.x, stop.y + 25);
    });
}

// -------------------- Route Animation --------------------
function showRouteAnimation() {
    const ctx = busNetworkCanvas.getContext('2d');
    let flashCount = 0;
    const flashInterval = setInterval(() => {
        drawBusNetwork();
        if (flashCount % 2 === 0) {
            ctx.fillStyle = 'rgba(255, 107, 107, 0.2)';
            ctx.fillRect(0, 0, busNetworkCanvas.width, busNetworkCanvas.height);
        }
        flashCount++;
        if (flashCount >= 4) {
            clearInterval(flashInterval);
            drawBusNetwork();
        }
    }, 300);
}

// -------------------- Event Listeners --------------------
// Find Route
findRouteBtn.addEventListener('click', () => {
    const source = document.getElementById('sourceStop').value;
    const destination = document.getElementById('destStop').value;
    if (!source || !destination) {
        alert('Please select both source and destination stops');
        return;
    }
    routeSegmentsDiv.style.display = 'block';
    drawBusNetwork();
    showRouteAnimation();
});

// Clear
clearBtn.addEventListener('click', () => {
    document.getElementById('sourceStop').value = '';
    document.getElementById('destStop').value = '';
    routeSegmentsDiv.style.display = 'none';
    drawBusNetwork();
});

// Option Buttons
optionButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        optionButtons.forEach(b => b.classList.remove('primary'));
        btn.classList.add('primary');
    });
});

// Tabs
tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const target = btn.dataset.tab;
        tabButtons.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(target).classList.add('active');
    });
});

// Music Cards
musicCards.forEach(card => {
    card.addEventListener('click', () => {
        musicCards.forEach(c => c.classList.remove('active'));
        card.classList.add('active');
    });
});

// Responsive
window.addEventListener('resize', drawBusNetwork);

// -------------------- Initial Draw --------------------
drawBusNetwork();
console.log('[v1] Bus Route Optimizer initialized successfully');
