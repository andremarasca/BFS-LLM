#!/bin/bash
# BFS-LLM Execution Script for Linux/Mac
# This script runs the concept tree expansion system

set -e

echo "========================================"
echo "BFS-LLM: Concept Tree Expansion"
echo "========================================"
echo

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Checking dependencies..."
pip install -q -r requirements.txt
echo

# Check if .env file exists and load it
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    echo "Or create a .env file based on .env.example"
    echo
    read -p "Press Enter to continue anyway..."
fi

# Run the main script
python main.py

echo
echo "========================================"
echo "Execution finished"
echo "========================================"
