#!/bin/bash
# Run the trading bot in the background with proper logging

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate venv if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Load .env if exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

case "$1" in
    start)
        if [ -f "bot.pid" ] && kill -0 $(cat bot.pid) 2>/dev/null; then
            echo "❌ Bot is already running (PID: $(cat bot.pid))"
            exit 1
        fi
        
        echo "🚀 Starting trading bot..."
        nohup python trading_bot.py --no-confirm >> bot.log 2>&1 &
        echo $! > bot.pid
        echo "✅ Bot started (PID: $(cat bot.pid))"
        echo "📄 Logs: tail -f bot.log"
        ;;
        
    stop)
        if [ -f "bot.pid" ]; then
            PID=$(cat bot.pid)
            if kill -0 $PID 2>/dev/null; then
                echo "⏹️  Stopping bot (PID: $PID)..."
                kill $PID
                rm bot.pid
                echo "✅ Bot stopped"
            else
                echo "⚠️  Bot not running (stale PID file)"
                rm bot.pid
            fi
        else
            echo "⚠️  No PID file found. Bot may not be running."
            # Try to find and kill anyway
            pkill -f "python.*trading_bot.py" && echo "✅ Killed trading_bot process"
        fi
        ;;
        
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
        
    status)
        if [ -f "bot.pid" ] && kill -0 $(cat bot.pid) 2>/dev/null; then
            echo "✅ Bot is running (PID: $(cat bot.pid))"
        else
            echo "❌ Bot is not running"
        fi
        ;;
        
    logs)
        tail -f bot.log
        ;;
        
    dashboard)
        echo "🖥️  Starting dashboard..."
        cd dashboard/backend
        python server.py
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|dashboard}"
        echo ""
        echo "Commands:"
        echo "  start     - Start the trading bot in background"
        echo "  stop      - Stop the trading bot"
        echo "  restart   - Restart the trading bot"
        echo "  status    - Check if bot is running"
        echo "  logs      - Follow the log file"
        echo "  dashboard - Start the web dashboard"
        exit 1
        ;;
esac