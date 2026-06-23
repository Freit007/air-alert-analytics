"""Main Streamlit entry point.

Uses st.navigation so the sidebar shows friendly page names
("Про ресурс", "Аналітика") instead of raw filenames.

Run with: streamlit run app.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

st.set_page_config(
    page_title="Air Alert Analytics — Ukraine",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("views/about.py", title="Про ресурс", icon="🚨", default=True),
    st.Page("views/analytics.py", title="Аналітика", icon="📊"),
]

st.navigation(pages).run()
