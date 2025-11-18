FPGA Self-Test System — Web UI + Python Integration

A lightweight FPGA Built-In Self-Test (BIST) monitoring tool featuring a modern web interface and a Python backend for real-time data simulation and analysis.

🚀 Features
🔹 Real-Time Monitoring

Live resistance tracking (per pin & overall)

Fault detection (≥100Ω spikes)

Pin health grid (8 pins with status colors)

1-second data updates

🔹 Python Backend

Simulated FPGA BIST data

Fault injection logic

Configurable degradation + resistance model

Easy integration with ML predictions

🔹 Modern Web UI

Clean, responsive dashboard

Real-time line charts

Fault timeline visualization

Interactive controls (Start / Stop / Reset)

JSON export support

🛠️ Tech Stack

Frontend: HTML, CSS, JavaScript

Charts: Chart.js (time-series)

Backend: Python

Design: Gradients, animations, glassmorphism

📁 Project Structure
├── index.html        # UI layout
├── styles.css        # Styling + animations
├── script.js         # Real-time dashboard logic
└── maincode.py       # Python simulation engine

▶️ How to Run
1. Run Python simulator
python maincode.py

2. Open UI

Simply open index.html in your browser.

📦 Future Add-Ons

WebSocket live FPGA data

ML predictions (RUL, failure probability)

Database logging

Alerts & notifications
