#!/bin/bash
exec .venv/bin/streamlit run app.py --server.headless true --server.port "${PORT:-8501}"
