#!/bin/bash
# Setup script for macOS launchd service

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         ToS Trading Bot - Service Setup                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Get current directory and username
CURRENT_DIR=$(pwd)
USERNAME=$(whoami)
PLIST_NAME="com.tos.tradingbot.plist"
PLIST_SRC="$CURRENT_DIR/$PLIST_NAME"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME"

# Check if plist exists
if [ ! -f "$PLIST_SRC" ]; then
    echo "❌ Error: $PLIST_NAME not found in current directory"
    exit 1
fi

# Update plist with correct paths
echo "📝 Configuring service..."
sed -i '' "s|/Users/YOUR_USERNAME|$HOME|g" "$PLIST_SRC"

# Detect Python path
PYTHON_PATH=$(which python3)
if [ -f "$CURRENT_DIR/venv/bin/python" ]; then
    PYTHON_PATH="$CURRENT_DIR/venv/bin/python"
    echo "   Using venv Python: $PYTHON_PATH"
else
    echo "   Using system Python: $PYTHON_PATH"
fi
sed -i '' "s|/usr/bin/python3|$PYTHON_PATH|g" "$PLIST_SRC"

# Copy to LaunchAgents
echo "📂 Installing service..."
mkdir -p "$HOME/Library/LaunchAgents"
cp "$PLIST_SRC" "$PLIST_DEST"

echo ""
echo "✅ Service installed!"
echo ""
echo "Commands:"
echo "─────────────────────────────────────────────────────────────"
echo "  Start bot:    launchctl load $PLIST_DEST"
echo "  Stop bot:     launchctl unload $PLIST_DEST"
echo "  View logs:    tail -f $CURRENT_DIR/bot.log"
echo "  View errors:  tail -f $CURRENT_DIR/bot_error.log"
echo "─────────────────────────────────────────────────────────────"
echo ""
echo "To start now:"
echo "  launchctl load $PLIST_DEST"
echo ""
