@echo off
echo Starting Focus Tracking Dashboard...
echo.
cd /d %~dp0
streamlit run dashboard.py
pause

