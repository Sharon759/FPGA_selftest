// FPGA Solder Joint Monitoring System - JavaScript
class FPGAMonitoringSystem {
    constructor() {
        this.isRunning = false;
        this.startTime = null;
        this.testDuration = 0;
        this.faultCount = 0;
        this.pinData = {};
        this.resistanceData = [];
        this.faultData = [];
        this.maxDataPoints = 100;
        this.lastSpikeTime = 0; // Track last spike time to prevent clustering
        
        // Initialize charts
        this.initCharts();
        this.initPinGrid();
        this.initEventListeners();
        
        // Start data simulation
        this.startDataSimulation();
        
        // Load dummy data initially
        this.loadDummyData();
    }

    initCharts() {
        // Resistance Chart
        const resistanceCtx = document.getElementById('resistanceChart').getContext('2d');
        this.resistanceChart = new Chart(resistanceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Resistance (Ω)',
                    data: [],
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            displayFormats: {
                                minute: 'HH:mm:ss'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Time',
                            color: '#e0e0e0'
                        },
                        ticks: {
                            color: '#b0b0b0'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Resistance (Ω)',
                            color: '#e0e0e0'
                        },
                        ticks: {
                            color: '#b0b0b0'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        min: 0,
                        max: 200
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#e0e0e0'
                        }
                    }
                },
                animation: {
                    duration: 750
                }
            }
        });

        // Fault Chart
        const faultCtx = document.getElementById('faultChart').getContext('2d');
        this.faultChart = new Chart(faultCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Faults per Minute',
                    data: [],
                    backgroundColor: 'rgba(100, 181, 246, 0.8)',
                    borderColor: '#64b5f6',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            displayFormats: {
                                minute: 'HH:mm'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Time',
                            color: '#e0e0e0'
                        },
                        ticks: {
                            color: '#b0b0b0'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Fault Count',
                            color: '#e0e0e0'
                        },
                        ticks: {
                            color: '#b0b0b0'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#e0e0e0'
                        }
                    }
                },
                animation: {
                    duration: 750
                }
            }
        });
    }

    initPinGrid() {
        const pinGrid = document.getElementById('pinGrid');
        const pinCount = 8;
        
        for (let i = 1; i <= pinCount; i++) {
            const pinCard = document.createElement('div');
            pinCard.className = 'pin-card healthy';
            pinCard.id = `pin-${i}`;
            
            pinCard.innerHTML = `
                <div class="pin-number">Pin ${i}</div>
                <div class="pin-resistance" id="pin-${i}-resistance">45.2 Ω</div>
                <div class="pin-status healthy" id="pin-${i}-status">Healthy</div>
            `;
            
            pinGrid.appendChild(pinCard);
            
            // Initialize pin data
            this.pinData[i] = {
                resistance: 45.2 + Math.random() * 10,
                status: 'healthy',
                faultCount: 0
            };
        }
    }

    initEventListeners() {
        document.getElementById('startTest').addEventListener('click', () => this.startTest());
        document.getElementById('stopTest').addEventListener('click', () => this.stopTest());
        document.getElementById('resetData').addEventListener('click', () => this.resetData());
        document.getElementById('exportData').addEventListener('click', () => this.exportData());
        
        document.getElementById('testDurationInput').addEventListener('change', (e) => {
            this.testDuration = parseInt(e.target.value) * 3600; // Convert to seconds
        });
        
        document.getElementById('thresholdInput').addEventListener('change', (e) => {
            this.faultThreshold = parseInt(e.target.value);
        });
    }

    startTest() {
        this.isRunning = true;
        this.startTime = new Date();
        this.faultCount = 0;
        this.resistanceData = [];
        this.faultData = [];
        
        // Update UI
        document.getElementById('connectionStatus').className = 'status-dot connected';
        document.getElementById('connectionText').textContent = 'Connected';
        
        // Clear charts
        this.resistanceChart.data.labels = [];
        this.resistanceChart.data.datasets[0].data = [];
        this.faultChart.data.labels = [];
        this.faultChart.data.datasets[0].data = [];
        
        this.updateCharts();
        console.log('Test started');
    }

    stopTest() {
        this.isRunning = false;
        document.getElementById('connectionStatus').className = 'status-dot';
        document.getElementById('connectionText').textContent = 'Disconnected';
        console.log('Test stopped');
    }

    resetData() {
        this.stopTest();
        this.faultCount = 0;
        this.resistanceData = [];
        this.faultData = [];
        this.testDuration = 0;
        
        // Reset UI
        document.getElementById('faultCount').textContent = '0';
        document.getElementById('testDuration').textContent = '00:00:00';
        document.getElementById('systemHealth').textContent = '100%';
        
        // Reset charts
        this.resistanceChart.data.labels = [];
        this.resistanceChart.data.datasets[0].data = [];
        this.faultChart.data.labels = [];
        this.faultChart.data.datasets[0].data = [];
        this.resistanceChart.update();
        this.faultChart.update();
        
        // Reset pin data
        for (let i = 1; i <= 8; i++) {
            this.pinData[i] = {
                resistance: 45.2 + Math.random() * 10,
                status: 'healthy',
                faultCount: 0
            };
            this.updatePinStatus(i);
        }
        
        console.log('Data reset');
    }

    exportData() {
        const data = {
            testDuration: this.testDuration,
            faultCount: this.faultCount,
            resistanceData: this.resistanceData,
            faultData: this.faultData,
            pinData: this.pinData,
            timestamp: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `fpga-test-data-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log('Data exported');
    }

    loadDummyData() {
        // Generate dummy resistance data for the last 30 minutes
        const now = new Date();
        const dummyResistanceData = [];
        const dummyFaultData = [];
        let lastSpikeTime = 0; // Track last spike time for dummy data
        
        for (let i = 30; i >= 0; i--) {
            const time = new Date(now.getTime() - (i * 60 * 1000)); // Every minute
            const baseResistance = 45 + (30 - i) * 0.5; // Gradual increase
            const noise = (Math.random() - 0.5) * 5;
            let resistance = Math.max(baseResistance + noise, 0);
            
            // Add some microsecond spikes to dummy data (spaced apart)
            const timeSinceLastSpike = (time.getTime() - lastSpikeTime) / 1000;
            const shouldCreateSpike = Math.random() < 0.2 && timeSinceLastSpike >= 5; // 20% chance, min 5s apart
            
            if (shouldCreateSpike) {
                const spikeHeight = 60 + Math.random() * 100; // 60-160Ω spike
                const spikeDuration = 0.001; // 1ms spike
                
                // Add spike point
                dummyResistanceData.push({
                    x: time,
                    y: Math.max(resistance + spikeHeight, 0)
                });
                
                // Add return to normal point
                const spikeEndTime = new Date(time.getTime() + spikeDuration);
                dummyResistanceData.push({
                    x: spikeEndTime,
                    y: Math.max(resistance, 0)
                });
                
                // Update last spike time
                lastSpikeTime = time.getTime();
                
                // Count as fault
                dummyFaultData.push({
                    x: time,
                    y: 1
                });
            } else {
                // Normal data point
                dummyResistanceData.push({
                    x: time,
                    y: resistance
                });
            }
        }
        
        // Populate charts with dummy data
        this.resistanceData = dummyResistanceData;
        this.faultData = dummyFaultData;
        
        // Update charts
        this.updateCharts();
        
        // Set initial UI values
        this.faultCount = dummyFaultData.reduce((sum, fault) => sum + fault.y, 0);
        document.getElementById('faultCount').textContent = this.faultCount;
        document.getElementById('systemHealth').textContent = '92%';
        
        // Update ML predictions with dummy data
        this.updateMLPredictions();
    }

    startDataSimulation() {
        setInterval(() => {
            if (this.isRunning) {
                this.simulateData();
                this.updateUI();
            }
        }, 1000); // Update every second
    }

    simulateData() {
        const now = new Date();
        const timeElapsed = (now - this.startTime) / 1000; // seconds
        
        // Simulate resistance changes with gradual degradation
        const baseResistance = 45;
        const degradationFactor = Math.min(timeElapsed / 3600, 2); // Max 2x degradation over time
        const noise = (Math.random() - 0.5) * 10;
        let resistance = baseResistance + (degradationFactor * 20) + noise;
        
        // Simulate microsecond-level spikes (BIST fault detection)
        // Only allow spikes if at least 3 seconds have passed since last spike
        const timeSinceLastSpike = (now.getTime() - this.lastSpikeTime) / 1000;
        const shouldCreateSpike = Math.random() < 0.08 && timeSinceLastSpike >= 3; // 8% chance, min 3s apart
        
        if (shouldCreateSpike) {
            // Create a very short-duration spike (simulating microsecond detection)
            const spikeHeight = 80 + Math.random() * 120; // 80-200Ω spike
            const spikeDuration = 0.001; // 1ms spike duration
            
            // Add the spike point
            this.resistanceData.push({
                x: now,
                y: Math.max(resistance + spikeHeight, 0)
            });
            
            // Add the return to normal point (very close in time)
            const spikeEndTime = new Date(now.getTime() + spikeDuration);
            this.resistanceData.push({
                x: spikeEndTime,
                y: Math.max(resistance, 0)
            });
            
            // Update last spike time
            this.lastSpikeTime = now.getTime();
            
            // Count this as a fault
            this.faultCount++;
            this.faultData.push({
                x: now,
                y: 1
            });
        } else {
            // Normal data point
            this.resistanceData.push({
                x: now,
                y: Math.max(resistance, 0)
            });
        }
        
        // Keep only recent data points
        if (this.resistanceData.length > this.maxDataPoints) {
            this.resistanceData.shift();
        }
        
        // Update pin data
        for (let pinId in this.pinData) {
            const pin = this.pinData[pinId];
            const pinResistance = baseResistance + (degradationFactor * 15) + (Math.random() - 0.5) * 8;
            pin.resistance = Math.max(pinResistance, 0);
            
            // Determine pin status
            if (pin.resistance > 100) {
                pin.status = 'critical';
                pin.faultCount++;
            } else if (pin.resistance > 80) {
                pin.status = 'warning';
            } else {
                pin.status = 'healthy';
            }
            
            this.updatePinStatus(pinId);
        }
        
        // Update test duration
        this.testDuration = Math.floor(timeElapsed);
    }

    updateUI() {
        // Update metrics
        document.getElementById('faultCount').textContent = this.faultCount;
        
        const hours = Math.floor(this.testDuration / 3600);
        const minutes = Math.floor((this.testDuration % 3600) / 60);
        const seconds = this.testDuration % 60;
        document.getElementById('testDuration').textContent = 
            `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        
        // Calculate system health
        const healthyPins = Object.values(this.pinData).filter(pin => pin.status === 'healthy').length;
        const totalPins = Object.keys(this.pinData).length;
        const healthPercentage = Math.round((healthyPins / totalPins) * 100);
        document.getElementById('systemHealth').textContent = `${healthPercentage}%`;
        
        // Update ML predictions
        this.updateMLPredictions();
        
        // Update charts
        this.updateCharts();
    }

    updatePinStatus(pinId) {
        const pin = this.pinData[pinId];
        const pinCard = document.getElementById(`pin-${pinId}`);
        const resistanceElement = document.getElementById(`pin-${pinId}-resistance`);
        const statusElement = document.getElementById(`pin-${pinId}-status`);
        
        // Update resistance display
        resistanceElement.textContent = `${pin.resistance.toFixed(1)} Ω`;
        
        // Update status
        statusElement.textContent = pin.status;
        statusElement.className = `pin-status ${pin.status}`;
        
        // Update card appearance
        pinCard.className = `pin-card ${pin.status}`;
    }

    updateMLPredictions() {
        // Simulate ML predictions based on current data
        const timeElapsed = this.testDuration / 3600; // hours
        const avgResistance = this.resistanceData.length > 0 ? 
            this.resistanceData.reduce((sum, point) => sum + point.y, 0) / this.resistanceData.length : 45;
        
        // RUL prediction (simplified model)
        const degradationRate = Math.max(0, (avgResistance - 45) / Math.max(timeElapsed, 1));
        const rulHours = Math.max(0, (100 - avgResistance) / Math.max(degradationRate, 0.1));
        document.getElementById('rulPrediction').textContent = `${rulHours.toFixed(1)} hours`;
        
        // Failure probability in next 15 hours
        const failureProb = Math.min(95, Math.max(5, (avgResistance - 45) * 2 + timeElapsed * 5));
        document.getElementById('failureProbability').textContent = `${Math.round(failureProb)}%`;
        
        // Degradation rate
        document.getElementById('degradationRate').textContent = `${degradationRate.toFixed(2)} Ω/hour`;
    }

    updateCharts() {
        // Update resistance chart
        if (this.resistanceData.length > 0) {
            this.resistanceChart.data.labels = this.resistanceData.map(point => point.x);
            this.resistanceChart.data.datasets[0].data = this.resistanceData.map(point => point.y);
            this.resistanceChart.update('none');
        }
        
        // Update fault chart (aggregate by minute)
        const faultByMinute = {};
        this.faultData.forEach(fault => {
            const minute = new Date(fault.x);
            minute.setSeconds(0, 0);
            const key = minute.getTime();
            faultByMinute[key] = (faultByMinute[key] || 0) + 1;
        });
        
        const faultLabels = Object.keys(faultByMinute).sort().map(key => new Date(parseInt(key)));
        const faultValues = Object.keys(faultByMinute).sort().map(key => faultByMinute[key]);
        
        this.faultChart.data.labels = faultLabels;
        this.faultChart.data.datasets[0].data = faultValues;
        this.faultChart.update('none');
    }
}

// Initialize the system when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.fpgaSystem = new FPGAMonitoringSystem();
    console.log('FPGA Monitoring System initialized');
});

// Add some utility functions for debugging
window.debugFPGA = {
    startTest: () => window.fpgaSystem.startTest(),
    stopTest: () => window.fpgaSystem.stopTest(),
    resetData: () => window.fpgaSystem.resetData(),
    getData: () => ({
        resistanceData: window.fpgaSystem.resistanceData,
        faultData: window.fpgaSystem.faultData,
        pinData: window.fpgaSystem.pinData
    })
};
