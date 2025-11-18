# FPGA Self-Test System (Web UI + Python)

A simple and modern FPGA Built-In Self-Test (BIST) monitoring tool with a clean web dashboard and a Python-based data simulator.

## Features
- Real-time resistance monitoring (per pin + overall)
- Automatic fault detection (≥100Ω spikes)
- Pin health grid with color indicators
- Python backend for data simulation
- Live charts (Chart.js) and fault timeline
- Start / Stop / Reset controls
- JSON data export

## Tech Stack
- Frontend: HTML, CSS, JavaScript  
- Backend: Python  
- Charts: Chart.js  

## Project Structure
├── index.html # Dashboard UI
├── styles.css # Styling
├── script.js # Frontend logic
└── maincode.py # Python simulator


## How to Run
1. Run the Python simulator:
2. Open the UI:



## Future Add-ons
- WebSocket live FPGA data  
- ML-based predictions (RUL, failure probability)  
- Database logging  
- Alert system  

