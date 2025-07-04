#!/bin/bash

# AI Tools Assignment - Streamlit App Runner
echo "🚀 Starting AI Tools Assignment Dashboard..."
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "streamlit_env" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv streamlit_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source streamlit_env/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model if not present
echo "📝 Setting up NLP models..."
python -m spacy download en_core_web_sm

# Run the application
echo "🌟 Launching Streamlit application..."
echo "📱 Open http://localhost:8501 in your browser"
echo "=========================================="

streamlit run app.py
