# Smart Traffic Signal RL Dashboard V2

## Run on Windows 11

From the project root:

```bat
venv\Scripts\python.exe -m pip install -r dashboard_requirements.txt
venv\Scripts\python.exe -m streamlit run dashboard_app.py
```

Or double-click `RUN_DASHBOARD.bat`.

## Dashboard features

- Live traffic intersection visualization
- Current NS/EW queues and signal phase
- Trained-policy action recommendation
- Step-by-step and 20-step simulation controls
- Live queue trajectory
- Training learning curves from saved episode rewards
- Comparison of the four implemented algorithms using the project's stored metrics
- Final policy and Q-value viewer

The dashboard reads existing project outputs and does not retrain or modify the four RL implementations.
