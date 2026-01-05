#!/bin/bash
# Trading Bot Manager - Multi-Symbol Support
# Usage: ./bot.sh start SPY
#        ./bot.sh start QQQ --paper
#        ./bot.sh stop SPY
#        ./bot.sh status

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Activate venv if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Load .env if exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Helper: Get safe symbol name (strip leading slash for futures)
get_safe_symbol() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/^\///'
}

# Helper: Get PID file path
get_pid_file() {
    local safe_symbol=$(get_safe_symbol "$1")
    echo "${safe_symbol}.pid"
}

# Helper: Get log file path
get_log_file() {
    local safe_symbol=$(get_safe_symbol "$1")
    echo "${safe_symbol}_bot.log"
}

# Helper: Check if a symbol's bot is running
is_running() {
    local pid_file=$(get_pid_file "$1")
    if [ -f "$pid_file" ] && kill -0 $(cat "$pid_file") 2>/dev/null; then
        return 0
    fi
    return 1
}

# Helper: List all running bots
list_running() {
    local found=0
    for pid_file in *.pid; do
        [ -f "$pid_file" ] || continue
        local symbol=$(basename "$pid_file" .pid | tr '[:lower:]' '[:upper:]')
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  ${GREEN}●${NC} $symbol (PID: $pid)"
            found=1
        else
            echo -e "  ${YELLOW}○${NC} $symbol (stale PID file)"
            rm "$pid_file" 2>/dev/null
        fi
    done
    if [ $found -eq 0 ]; then
        echo -e "  ${RED}No bots running${NC}"
    fi
}

case "$1" in
    start)
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Please specify a symbol${NC}"
            echo "Usage: $0 start SYMBOL [--paper]"
            echo "Examples:"
            echo "  $0 start SPY"
            echo "  $0 start QQQ --paper"
            echo "  $0 start \"/BTC\" --paper"
            exit 1
        fi
        
        SYMBOL="$2"
        EXTRA_ARGS="${@:3}"  # Capture --paper and other args
        PID_FILE=$(get_pid_file "$SYMBOL")
        LOG_FILE=$(get_log_file "$SYMBOL")
        
        if is_running "$SYMBOL"; then
            echo -e "${RED}❌ $SYMBOL bot is already running (PID: $(cat $PID_FILE))${NC}"
            exit 1
        fi
        
        echo -e "${BLUE}🚀 Starting $SYMBOL trading bot...${NC}"
        nohup python trading_bot.py --symbol "$SYMBOL" --no-confirm $EXTRA_ARGS >> "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        sleep 1
        
        if is_running "$SYMBOL"; then
            echo -e "${GREEN}✅ $SYMBOL bot started (PID: $(cat $PID_FILE))${NC}"
            echo -e "📄 Logs: tail -f $LOG_FILE"
        else
            echo -e "${RED}❌ Failed to start $SYMBOL bot. Check $LOG_FILE${NC}"
            rm "$PID_FILE" 2>/dev/null
            exit 1
        fi
        ;;
        
    stop)
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Please specify a symbol or use 'stop-all'${NC}"
            echo "Usage: $0 stop SYMBOL"
            echo "       $0 stop-all"
            exit 1
        fi
        
        SYMBOL="$2"
        PID_FILE=$(get_pid_file "$SYMBOL")
        
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo -e "${YELLOW}⏹️  Stopping $SYMBOL bot (PID: $PID)...${NC}"
                kill "$PID"
                sleep 2
                if kill -0 "$PID" 2>/dev/null; then
                    echo -e "${YELLOW}   Sending SIGKILL...${NC}"
                    kill -9 "$PID" 2>/dev/null
                fi
                rm "$PID_FILE"
                echo -e "${GREEN}✅ $SYMBOL bot stopped${NC}"
            else
                echo -e "${YELLOW}⚠️  $SYMBOL bot not running (stale PID file)${NC}"
                rm "$PID_FILE"
            fi
        else
            echo -e "${YELLOW}⚠️  No PID file for $SYMBOL. Bot may not be running.${NC}"
        fi
        ;;
        
    stop-all)
        echo -e "${YELLOW}⏹️  Stopping all bots...${NC}"
        for pid_file in *.pid; do
            [ -f "$pid_file" ] || continue
            symbol=$(basename "$pid_file" .pid | tr '[:lower:]' '[:upper:]')
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "   Stopping $symbol (PID: $pid)..."
                kill "$pid"
            fi
            rm "$pid_file" 2>/dev/null
        done
        sleep 2
        # Force kill any stragglers
        pkill -f "python.*trading_bot.py" 2>/dev/null
        echo -e "${GREEN}✅ All bots stopped${NC}"
        ;;
        
    restart)
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Please specify a symbol${NC}"
            echo "Usage: $0 restart SYMBOL [--paper]"
            exit 1
        fi
        SYMBOL="$2"
        EXTRA_ARGS="${@:3}"
        $0 stop "$SYMBOL"
        sleep 2
        $0 start "$SYMBOL" $EXTRA_ARGS
        ;;
        
    status)
        echo -e "${BLUE}📊 Trading Bot Status${NC}"
        echo "────────────────────────"
        list_running
        echo ""
        ;;
        
    logs)
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Please specify a symbol${NC}"
            echo "Usage: $0 logs SYMBOL"
            echo ""
            echo "Available logs:"
            ls -la *_bot.log 2>/dev/null || echo "  No log files found"
            exit 1
        fi
        
        LOG_FILE=$(get_log_file "$2")
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo -e "${RED}❌ Log file not found: $LOG_FILE${NC}"
        fi
        ;;
        
    dashboard)
        echo -e "${BLUE}🖥️  Starting dashboard...${NC}"
        cd dashboard/backend
        python server.py
        ;;
        
    *)
        echo -e "${BLUE}Trading Bot Manager - Multi-Symbol${NC}"
        echo "Usage: $0 {start|stop|stop-all|restart|status|logs|dashboard}"
        echo ""
        echo -e "${GREEN}Commands:${NC}"
        echo "  start SYMBOL [--paper]  - Start bot for SYMBOL"
        echo "  stop SYMBOL             - Stop bot for SYMBOL"
        echo "  stop-all                - Stop all running bots"
        echo "  restart SYMBOL          - Restart bot for SYMBOL"
        echo "  status                  - Show all running bots"
        echo "  logs SYMBOL             - Follow log file for SYMBOL"
        echo "  dashboard               - Start the web dashboard"
        echo ""
        echo -e "${GREEN}Examples:${NC}"
        echo "  $0 start SPY                 # Start SPY live"
        echo "  $0 start QQQ --paper         # Start QQQ paper trading"
        echo "  $0 start \"/BTC\" --paper      # Start BTC futures paper"
        echo "  $0 stop SPY                  # Stop SPY bot"
        echo "  $0 stop-all                  # Stop all bots"
        echo "  $0 status                    # Show running bots"
        echo "  $0 logs SPY                  # Follow SPY logs"
        exit 1
        ;;
esac