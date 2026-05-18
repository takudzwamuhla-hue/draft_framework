# AI Risk Management Dashboard — Beverage Line
### NUST MEng Research · Mark Mbewe (N02534127N) · Supervisor: Eng T. Muhla

## Overview
A Streamlit dashboard implementing the AI-powered risk management framework
from the dissertation "Development of an AI Powered Risk Management Framework
for Zimbabwean Manufacturing Plants".

Features:
- **Real-time simulation** cycling through the 500-observation pilot dataset with
  Gaussian noise to mimic live sensor variation
- **Two-layer risk engine** — rule-based status classification (100% accurate on
  audited labels) + telemetry early-warning scoring (91.4% accuracy)
- **Explainable risk decisions** showing which node triggered the alert and why
- **Claude AI recommendations** per machine node via the Anthropic API
- **Live sparkline charts** for each of the 6 machine nodes
- **Event log** with timestamped entries for post-shift audit
- **Confusion matrix** from pilot validation
- **Implementation roadmap** based on Chapter 6 recommendations
- **Scenario selector** to jump straight to Normal / Film Fault / Recovery states

## Machine Nodes
| # | Machine | Brand | Primary KPI |
|---|---------|-------|-------------|
| 1 | KHS Innofill Glass Filler | KHS Group | Fill speed (bph) |
| 2 | Krones Checkmat Inspector | Krones AG | Reject rate (%) |
| 3 | Sidel SBO Universal Moulder | Sidel | Cycle time (s) |
| 4 | Tetra Pak Pasteurizer | Tetra Pak | Outlet temp (°C) |
| 5 | Krones Variopac Wrapper | Krones AG | Film remaining (%) |
| 6 | Heuft SPECTRUM II Conveyor | Heuft | Vibration (mm/s) |

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app opens at http://localhost:8501

## Controls (sidebar)
- **▶ Step** — advance one simulation cycle manually
- **⟳ Reset** — restart from cycle 0, clear history and logs
- **Auto-cycle** — toggle automatic 10-second refresh
- **Jump to Scenario** — preload Normal / Film Fault / Recovery observation
- **Ask Claude** — request an AI recommendation for the selected machine node
  (requires Anthropic API access — the API key must be configured in the
  environment where Streamlit is served; within claude.ai Artifacts the key is
  injected automatically)

## Dataset
`MBEWE_DS_v1` — 500 time-stamped observations across 6 interdependent
beverage-line nodes. Distribution: 72% Low risk, 18% Medium, 10% High.

## Risk Engine Logic
```
Any Fault node            → HIGH RISK
Any Warning / Idle node   → MEDIUM RISK
Telemetry score > 20 %    → EARLY WARNING
All nodes nominal         → LOW RISK
```

Telemetry score = weighted count of parameters outside thresholds / total
possible deviations × 100. Score ≥ 20% triggers an early warning before
a formal fault state is reached.
