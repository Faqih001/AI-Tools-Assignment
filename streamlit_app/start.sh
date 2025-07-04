#!/bin/bash

# AI Tools Assignment - Streamlit App Deployment Script

echo "🚀 AI Tools Assignment - Streamlit Deployment"
echo "=============================================="

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Setup models
echo "🔧 Setting up models..."
python setup.py

# Start the application
echo "🎉 Starting Streamlit application..."
streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
