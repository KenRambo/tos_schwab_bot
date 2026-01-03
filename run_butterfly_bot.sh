#!/bin/bash
#
# Butterfly Bot Daemon Runner
#
# Usage:
#   ./run_butterfly_bot.sh start [--paper] [--equity 10000]
#   ./run_butterfly_bot.sh stop
#   ./run_butterfly_bot.sh status
#   ./run_butterfly_bot.sh logs
#   ./run_butterfly_bot.sh tail
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_SCRIPT="$SCRIPT_DIR/butterfly_bot_prod_fixed.py"
PID_FILE="$SCRIPT_DIR/butterfly_bot.pid"
LOG_FILE="$SCRIPT_DIR/butterfly_bot.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    else
        echo ""
    fi
}

is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

start_bot() {
    if is_running; then
        echo -e "${YELLOW}⚠️  Bot is already running (PID: $(get_pid))${NC}"
        exit 1
    fi
    
    # Remove stale PID file
    rm -f "$PID_FILE"
    
    # Pass through any additional arguments (--paper, --equity, etc.)
    shift  # Remove 'start' from args
    
    echo -e "${GREEN}🦋 Starting Butterfly Bot...${NC}"
    
    # Start bot in background, redirect output to log
    nohup python3 "$BOT_SCRIPT" "$@" >> "$LOG_FILE" 2>&1 &
    local pid=$!
    
    # Save PID
    echo $pid > "$PID_FILE"
    
    # Wait a moment and verify it started
    sleep 2
    
    if is_running; then
        echo -e "${GREEN}✓ Bot started successfully (PID: $pid)${NC}"
        echo -e "  Log file: $LOG_FILE"
        echo -e "  Use './run_butterfly_bot.sh tail' to watch logs"
    else
        echo -e "${RED}✗ Bot failed to start. Check logs:${NC}"
        tail -20 "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

stop_bot() {
    if ! is_running; then
        echo -e "${YELLOW}⚠️  Bot is not running${NC}"
        rm -f "$PID_FILE"
        exit 0
    fi
    
    local pid=$(get_pid)
    echo -e "${YELLOW}🛑 Stopping Butterfly Bot (PID: $pid)...${NC}"
    
    # Send SIGTERM for graceful shutdown
    kill -TERM "$pid" 2>/dev/null
    
    # Wait up to 10 seconds for graceful shutdown
    local count=0
    while is_running && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
        echo -n "."
    done
    echo ""
    
    # Force kill if still running
    if is_running; then
        echo -e "${RED}Force killing...${NC}"
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi
    
    rm -f "$PID_FILE"
    
    if is_running; then
        echo -e "${RED}✗ Failed to stop bot${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ Bot stopped${NC}"
    fi
}

status_bot() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "🦋 BUTTERFLY BOT STATUS"
    echo "═══════════════════════════════════════════════════════════"
    
    if is_running; then
        local pid=$(get_pid)
        echo -e "Status:  ${GREEN}RUNNING${NC}"
        echo -e "PID:     $pid"
        
        # Show process info
        echo ""
        echo "Process:"
        ps -p "$pid" -o pid,ppid,%cpu,%mem,etime,command --no-headers 2>/dev/null | head -1
        
        # Show recent log entries
        echo ""
        echo "Recent activity (last 10 lines):"
        echo "─────────────────────────────────────────────────────────"
        tail -10 "$LOG_FILE" 2>/dev/null || echo "  (no logs yet)"
        
        # Count today's signals/trades from log
        local today=$(date +%Y-%m-%d)
        local signals=$(grep -c "SIGNAL DETECTED" "$LOG_FILE" 2>/dev/null | grep "$today" || echo "0")
        local trades=$(grep -c "Position created" "$LOG_FILE" 2>/dev/null | grep "$today" || echo "0")
        
    else
        echo -e "Status:  ${RED}STOPPED${NC}"
        
        # Check for stale PID file
        if [ -f "$PID_FILE" ]; then
            echo -e "${YELLOW}(stale PID file found, cleaning up)${NC}"
            rm -f "$PID_FILE"
        fi
    fi
    
    echo ""
    echo "Log file: $LOG_FILE"
    if [ -f "$LOG_FILE" ]; then
        local log_size=$(du -h "$LOG_FILE" | cut -f1)
        local log_lines=$(wc -l < "$LOG_FILE")
        echo "Log size: $log_size ($log_lines lines)"
    fi
    echo "═══════════════════════════════════════════════════════════"
    echo ""
}

show_logs() {
    if [ -f "$LOG_FILE" ]; then
        less +G "$LOG_FILE"
    else
        echo "No log file found at $LOG_FILE"
    fi
}

tail_logs() {
    if [ -f "$LOG_FILE" ]; then
        echo -e "${GREEN}📜 Tailing log file (Ctrl+C to stop)...${NC}"
        echo ""
        tail -f "$LOG_FILE"
    else
        echo "No log file found at $LOG_FILE"
        echo "Start the bot first with: ./run_butterfly_bot.sh start"
    fi
}

restart_bot() {
    echo -e "${YELLOW}🔄 Restarting Butterfly Bot...${NC}"
    stop_bot
    sleep 2
    start_bot "$@"
}

clear_logs() {
    if is_running; then
        echo -e "${YELLOW}⚠️  Bot is running. Stop it first before clearing logs.${NC}"
        exit 1
    fi
    
    if [ -f "$LOG_FILE" ]; then
        # Backup old log
        local backup="$LOG_FILE.$(date +%Y%m%d_%H%M%S).bak"
        mv "$LOG_FILE" "$backup"
        echo -e "${GREEN}✓ Logs backed up to: $backup${NC}"
    else
        echo "No log file to clear"
    fi
}

usage() {
    echo ""
    echo "🦋 Butterfly Bot Daemon Runner"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start [opts]  Start the bot in background"
    echo "                Options:"
    echo "                  --paper         Paper trading mode"
    echo "                  --equity N      Starting equity (paper mode)"
    echo ""
    echo "  stop          Stop the running bot"
    echo "  restart       Restart the bot"
    echo "  status        Show bot status and recent logs"
    echo "  logs          View full log file (less)"
    echo "  tail          Follow log file in real-time"
    echo "  clear-logs    Backup and clear log file"
    echo ""
    echo "Examples:"
    echo "  $0 start                     # Start live trading"
    echo "  $0 start --paper             # Start paper trading"
    echo "  $0 start --paper --equity 5000"
    echo "  $0 status"
    echo "  $0 tail"
    echo "  $0 stop"
    echo ""
}

# Main command handler
case "${1:-}" in
    start)
        start_bot "$@"
        ;;
    stop)
        stop_bot
        ;;
    restart)
        restart_bot "$@"
        ;;
    status)
        status_bot
        ;;
    logs)
        show_logs
        ;;
    tail)
        tail_logs
        ;;
    clear-logs)
        clear_logs
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
esac