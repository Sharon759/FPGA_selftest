# FPGA Solder Joint Monitoring System - Web UI

A real-time web interface for monitoring FPGA-based solder joint health using Built-In Self-Test (BIST) technology and machine learning predictions.

## 🚀 Features

### Real-time Monitoring
- **Live Resistance Tracking**: Real-time graphs showing solder joint resistance over time
- **Fault Detection**: Visual timeline of resistance spikes ≥ 100Ω
- **Pin Status Grid**: Individual monitoring of 8 test pins with health indicators
- **System Health**: Overall health percentage based on pin status

### Machine Learning Predictions
- **Remaining Useful Life (RUL)**: Predicted time until joint failure
- **Failure Probability**: Likelihood of failure within 15 hours
- **Degradation Rate**: Rate of resistance increase over time
- **Confidence Indicators**: Visual confidence levels for each prediction

### Interactive Controls
- **Test Management**: Start/stop/reset test operations
- **Data Export**: Export test data as JSON for analysis
- **Configurable Parameters**: Adjustable test duration and fault thresholds
- **Real-time Updates**: Live data refresh every second

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Charts**: Chart.js with time-series support
- **Styling**: Modern CSS with gradients and animations
- **Responsive Design**: Mobile-friendly interface

## 📁 Project Structure

```
├── index.html          # Main HTML structure
├── styles.css          # CSS styling and animations
├── script.js           # JavaScript functionality and data simulation
└── README.md           # Project documentation
```

## 🚀 Getting Started

1. **Clone or download** the project files
2. **Open** `index.html` in a modern web browser
3. **Click "Start Test"** to begin monitoring simulation
4. **Observe** real-time data updates and ML predictions

## 🔧 Usage

### Starting a Test
1. Click the **"Start Test"** button
2. The system will begin simulating FPGA data
3. Watch real-time resistance graphs and fault detection
4. Monitor individual pin status in the grid below

### Monitoring Data
- **Resistance Chart**: Shows live resistance values over time
- **Fault Timeline**: Displays fault occurrences per minute
- **Pin Grid**: Individual pin health with color-coded status
- **ML Predictions**: Real-time failure predictions and RUL estimates

### Exporting Data
1. Click **"Export Data"** to download test results
2. Data includes resistance history, fault logs, and pin status
3. JSON format for easy integration with analysis tools

## 📊 Data Simulation

The system includes a comprehensive data simulator that mimics real FPGA BIST behavior:

- **Gradual Degradation**: Resistance increases over time
- **Random Faults**: Simulated resistance spikes
- **Pin Variation**: Different degradation rates per pin
- **Realistic Timing**: 1-second update intervals

## 🎨 UI Features

### Modern Design
- **Glassmorphism**: Frosted glass effects with backdrop blur
- **Gradient Backgrounds**: Beautiful color transitions
- **Smooth Animations**: Hover effects and transitions
- **Responsive Layout**: Works on desktop and mobile

### Color Coding
- **Green**: Healthy pins and good predictions
- **Yellow**: Warning status and moderate risk
- **Red**: Critical status and high failure probability
- **Blue**: System information and controls

## 🔮 Machine Learning Integration

The UI is designed to easily integrate with your actual ML models:

### Data Format
```javascript
{
  "resistanceData": [{"x": timestamp, "y": resistance}],
  "faultData": [{"x": timestamp, "y": faultCount}],
  "pinData": {
    "1": {"resistance": 45.2, "status": "healthy", "faultCount": 0}
  }
}
```

### API Endpoints (Future Integration)
- `POST /api/start-test` - Start monitoring session
- `GET /api/current-data` - Get latest sensor data
- `POST /api/ml-predict` - Get ML predictions
- `GET /api/export-data` - Download test results

## 🚀 Future Enhancements

- **WebSocket Integration**: Real-time data from actual FPGA
- **Database Storage**: Persistent data logging
- **Advanced ML Models**: Integration with trained LSTM/CNN models
- **Alert System**: Email/SMS notifications for critical failures
- **Historical Analysis**: Long-term trend analysis and reporting

## 🐛 Debugging

Use the browser console to access debugging functions:

```javascript
// Start test programmatically
window.debugFPGA.startTest();

// Stop test
window.debugFPGA.stopTest();

// Get current data
window.debugFPGA.getData();

// Reset all data
window.debugFPGA.resetData();
```

## 📱 Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 🤝 Contributing

This is a mini-project template. Feel free to extend and modify for your specific FPGA monitoring needs!

## 📄 License

Open source - feel free to use and modify for your projects.
