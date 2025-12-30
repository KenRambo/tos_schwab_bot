#!/bin/bash
# Launch the Trading Bot Dashboard

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         ToS Trading Bot - Dashboard Launcher                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "config.py" ]; then
    echo "Error: Please run this from the tos_schwab_bot directory"
    exit 1
fi

# Check for required packages
python3 -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install fastapi uvicorn websockets
fi

# Export environment variables if .env exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "Starting dashboard server..."
echo ""
echo "📊 Dashboard: http://localhost:8000"
echo "📚 API Docs:  http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the server
cd dashboard/backend
python3 server.py
