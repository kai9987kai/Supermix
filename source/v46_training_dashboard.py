import argparse
import json
import os
import time
import datetime
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request

app = Flask(__name__)

# Design System: Deep Space / Glassmorphism
HTML = """<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <title>Omni-Collective V46 Frontier Dashboard</title>
    <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
    <style>
        :root {
            --bg: #030712;
            --card-bg: rgba(17, 24, 39, 0.7);
            --border: rgba(255, 255, 255, 0.1);
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.5);
            --text-main: #f3f4f6;
            --text-dim: #9ca3af;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }

        * { box-sizing: border-box; }
        body {
            margin: 0;
            background: var(--bg);
            background-image: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #030712 100%);
            color: var(--text-main);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2.5rem;
            animation: fadeInDown 0.8s ease-out;
        }

        .brand-block h1 {
            margin: 0;
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            background: linear-gradient(to right, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-block p {
            margin: 0.2rem 0 0;
            color: var(--text-dim);
            font-size: 0.9rem;
        }

        .status-badge {
            background: var(--card-bg);
            border: 1px solid var(--border);
            padding: 0.5rem 1rem;
            border-radius: 99px;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-size: 0.85rem;
            font-weight: 600;
            backdrop-filter: blur(8px);
        }

        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 10px var(--success);
        }

        .dot.pulse { animation: pulse 2s infinite; }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            padding: 1.5rem;
            border-radius: 1.25rem;
            backdrop-filter: blur(12px);
            transition: transform 0.3s ease, border-color 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .card:hover {
            transform: translateY(-4px);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .card-label {
            color: var(--text-dim);
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .card-value {
            font-size: 1.75rem;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }

        .card-subtext {
            color: var(--text-dim);
            font-size: 0.85rem;
            margin-top: 0.4rem;
        }

        .big-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
        }

        .chart-container {
            height: 400px;
        }

        .log-container {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .log-window {
            flex-grow: 1;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 0.75rem;
            padding: 1rem;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.8rem;
            color: #d1d5db;
            overflow-y: auto;
            margin-top: 0.5rem;
            border: 1px solid var(--border);
            max-height: 350px;
        }

        .progress-bar-wrap {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 4px;
            margin-top: 1rem;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #60a5fa);
            box-shadow: 0 0 10px var(--accent-glow);
            transition: width 0.5s ease;
        }

        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .reasoning-viz {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .viz-block {
            flex: 1;
            padding: 0.75rem;
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            text-align: center;
        }

        .viz-block.active {
            border-color: var(--accent);
            background: rgba(59, 130, 246, 0.05);
        }

        .viz-label { font-size: 0.7rem; color: var(--text-dim); margin-bottom: 0.2rem; }
        .viz-value { font-size: 0.9rem; font-weight: 700; }

        @media (max-width: 900px) {
            .big-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class='container'>
        <header>
            <div class='brand-block'>
                <h1>Universe <span style='font-weight:300'>V46 Frontier</span></h1>
                <p>Advanced Agentic Reasoning Training Pipeline</p>
            </div>
            <div class='status-badge'>
                <div id='statusDot' class='dot pulse'></div>
                <span id='statusText'>POLLING...</span>
            </div>
        </header>

        <div class='grid'>
            <div class='card'>
                <div class='card-label'>Epoch / Step</div>
                <div class='card-value'><span id='epoch'>-</span> / <span id='step'>-</span></div>
                <div class='card-subtext'>Total Batches: <span id='totalSteps'>-</span></div>
                <div class='progress-bar-wrap'>
                    <div id='progressBar' class='progress-bar-fill' style='width: 0%'></div>
                </div>
            </div>
            <div class='card'>
                <div class='card-label'>Average Loss</div>
                <div id='loss' class='card-value' style='color: var(--accent)'>-.----</div>
                <div class='card-subtext'>Balance: <span id='balanceLoss'>-.----</span></div>
            </div>
            <div class='card'>
                <div class='card-label'>Learning Rate</div>
                <div id='lr' class='card-value'>-.----</div>
                <div class='card-subtext'>Decay Mode: Linear Warmup</div>
            </div>
            <div class='card'>
                <div class='card-label'>ETA Remaining</div>
                <div id='eta' class='card-value' style='color: var(--warning)'>--:--:--</div>
                <div class='card-subtext'>Elapsed: <span id='elapsed'>-</span></div>
            </div>
        </div>

        <div class='big-grid'>
            <div class='card'>
                <div class='card-label'>Training Dynamics (Loss Curve)</div>
                <div class='chart-container'>
                    <canvas id='lossChart'></canvas>
                </div>
            </div>
            <div class='log-container'>
                <div class='card' style='height: 100%;'>
                    <div class='card-label'>Internal Monologue (Logs)</div>
                    <div id='logWindow' class='log-window'>
                        Connecting to monitor server...
                    </div>
                    <div class='card-label' style='margin-top: 1.5rem'>Reasoning Scaffold</div>
                    <div class='reasoning-viz'>
                        <div id='vizGot' class='viz-block'>
                            <div class='viz-label'>GoT</div>
                            <div class='viz-value'>400 Rows</div>
                        </div>
                        <div id='vizCcot' class='viz-block'>
                            <div class='viz-label'>C-CoT</div>
                            <div class='viz-value'>400 Rows</div>
                        </div>
                        <div id='vizDiv' class='viz-block'>
                            <div class='viz-label'>Diversity</div>
                            <div class='viz-value'>24 Rows</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let lossHistory = [];
        let labels = [];
        const ctx = document.getElementById('lossChart').getContext('2d');
        const lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Avg Train Loss',
                    data: lossHistory,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { 
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#9ca3af', font: { family: 'JetBrains Mono' } }
                    }
                }
            }
        });

        function formatTime(seconds) {
            if (!seconds) return '--:--:--';
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
        }

        async function update() {
            try {
                const res = await fetch('/api/metrics');
                const data = await res.json();
                
                if (data.status === 'complete' || data.status === 'done') {
                    document.getElementById('statusText').textContent = 'COMPLETE';
                    document.getElementById('statusDot').className = 'dot';
                    document.getElementById('statusDot').style.background = 'var(--success)';
                    document.getElementById('loss').textContent = data.stage2_best_score ? data.stage2_best_score.toFixed(4) : 'N/A';
                    document.getElementById('loss').parentElement.querySelector('.card-label').textContent = 'Final Score';
                    document.getElementById('eta').textContent = '00:00:00';
                    document.getElementById('progressBar').style.width = '100%';
                    return;
                }
                
                document.getElementById('epoch').textContent = data.epoch || '1';
                document.getElementById('step').textContent = data.batch_index || '0';
                document.getElementById('totalSteps').textContent = data.total_batches || '0';
                document.getElementById('loss').textContent = data.avg_train_loss ? data.avg_train_loss.toFixed(4) : '-.----';
                document.getElementById('balanceLoss').textContent = data.avg_balance_loss ? data.avg_balance_loss.toFixed(4) : '-.----';
                document.getElementById('lr').textContent = data.lr ? data.lr.toExponential(2) : '-.----';
                document.getElementById('eta').textContent = formatTime(data.eta_seconds);
                document.getElementById('elapsed').textContent = formatTime(data.elapsed_seconds);
                
                const progress = (data.batch_index / data.total_batches) * 100;
                document.getElementById('progressBar').style.width = progress + '%';
                
                document.getElementById('statusText').textContent = data.status.toUpperCase().replace('_', ' ');
                if (data.status === 'stage_running') {
                    document.getElementById('statusDot').className = 'dot pulse';
                    document.getElementById('statusDot').style.background = 'var(--success)';
                } else {
                    document.getElementById('statusDot').className = 'dot';
                    document.getElementById('statusDot').style.background = 'var(--warning)';
                }

                // Update Chart
                if (data.batch_index && (!lossHistory.length || lossHistory[lossHistory.length-1] !== data.avg_train_loss)) {
                    lossHistory.push(data.avg_train_loss);
                    labels.push(data.batch_index);
                    if (lossHistory.length > 100) {
                        lossHistory.shift();
                        labels.shift();
                    }
                    lossChart.update('none');
                }

                // Mock Logs (Tail)
                const logWindow = document.getElementById('logWindow');
                const ts = new Date().toLocaleTimeString();
                const newLog = `[${ts}] Epoch ${data.epoch} | Batch ${data.batch_index} | Loss: ${data.avg_train_loss?.toFixed(4)}\\n`;
                if (!logWindow.textContent.includes(newLog)) {
                    logWindow.textContent += newLog;
                    logWindow.scrollTop = logWindow.scrollHeight;
                }

                // Reasoning Viz
                if (data.stage === 'stage1') {
                    document.getElementById('vizGot').className = 'viz-block active';
                    document.getElementById('vizCcot').className = 'viz-block active';
                }

            } catch (e) {
                console.error(e);
                document.getElementById('statusText').textContent = 'OFFLINE';
                document.getElementById('statusDot').style.background = 'var(--danger)';
            }
        }

        setInterval(update, 2000);
        update();
    </script>
</body>
</html>"""

STATE_PATH = Path("output/v46_train_artifacts/omni_collective_v46_train_state.json")

class Monitor:
    def __init__(self):
        self.history: List[float] = []
        self.last_update = 0

    def get_metrics(self) -> Dict[str, Any]:
        if not STATE_PATH.exists():
            return {"status": "waiting", "message": "State file not found"}
        
        try:
            with open(STATE_PATH, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            return {"status": "error", "message": str(e)}

monitor = Monitor()

@app.get('/')
def index():
    return HTML

@app.get('/api/metrics')
def api_metrics():
    return jsonify(monitor.get_metrics())

def main():
    parser = argparse.ArgumentParser(description="V46 Training Dashboard")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"Launching V46 Frontier Dashboard on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()
