"""
16_dashboard_server.py

BlazeMetrics Example – Dashboard Server (FastAPI + Dash UI)
----------------------------------------------------------
Launches the BlazeMetrics dashboard interface for real-time metrics monitoring, guardrail checks, agent/code evals, and analytics visualization.
- Backend: FastAPI
- Frontend: Dash (multi-tab browser UI)

Usage:
- Run this script to start a web UI at http://localhost:8000/dashboard
- Use for monitoring, manual evaluation, QA workflows, and demos
"""
from blazemetrics.dashboard import run_dashboard

# This will start a FastAPI backend with live Dash UI
if __name__ == '__main__':
    run_dashboard()
