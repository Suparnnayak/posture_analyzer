#!/bin/bash
echo "Starting Focus Tracking Dashboard..."
cd "$(dirname "$0")"
streamlit run dashboard.py

