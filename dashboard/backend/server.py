"""
Trading Bot Dashboard - FastAPI Backend

Provides REST API and WebSocket for real-time updates.
"""
import os
import sys
import json
import asyncio
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import BotConfig, config
from schwab_auth import SchwabAuth
from schwab_client import SchwabClient
from signal_detector import SignalDetector, Bar, Direction
from position_manager import PositionManager

# ==================== MODELS ====================

class ConfigUpdate(BaseModel):
    """Configuration update request"""
    paper_trading: Optional[bool] = None
    contracts: Optional[int] = None
    max_daily_trades: Optional[int] = None
    max_days_to_expiry: Optional[int] = None
    enable_stop_loss: Optional[bool] = None
    stop_loss_percent: Optional[float] = None
    stop_loss_dollars: Optional[float] = None
    enable_take_profit: Optional[bool] = None
    take_profit_percent: Optional[float] = None
    take_profit_dollars: Optional[float] = None
    enable_trailing_stop: Optional[bool] = None
    trailing_stop_percent: Optional[float] = None
    trailing_stop_activation: Optional[float] = None
    use_or_bias_filter: Optional[bool] = None
    rth_only: Optional[bool] = None
    signal_cooldown_bars: Optional[int] = None


class ManualSignal(BaseModel):
    """Manual signal trigger"""
    direction: str  # "long" or "short"


# ==================== APP SETUP ====================

app = FastAPI(title="ToS Trading Bot Dashboard", version="1.0.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GLOBAL STATE ====================

class BotState:
    """Global bot state accessible by API"""
    def __init__(self):
        self.client: Optional[SchwabClient] = None
        self.auth: Optional[SchwabAuth] = None
        self.detector: Optional[SignalDetector] = None
        self.position_manager: Optional[PositionManager] = None
        self.config = BotConfig()
        self.is_running = False
        self.connected_websockets: List[WebSocket] = []
        
        # Data storage
        self.bars: List[Dict] = []
        self.signals: List[Dict] = []
        self.current_price: float = 0
        self.last_update: Optional[datetime] = None
        
        # Initialize on startup
        self._initialize()
    
    def _initialize(self):
        """Initialize Schwab connection"""
        try:
            self.auth = SchwabAuth(
                app_key=os.getenv("SCHWAB_APP_KEY", ""),
                app_secret=os.getenv("SCHWAB_APP_SECRET", ""),
                redirect_uri=os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1:8182/callback"),
                token_file="schwab_tokens.json"
            )
            
            if self.auth.is_authenticated:
                self.auth.refresh_access_token()
                self.client = SchwabClient(self.auth)
                
            # Initialize detector
            self.detector = SignalDetector(
                length_period=self.config.signal.length_period,
                volume_threshold=self.config.signal.volume_threshold,
                use_relaxed_volume=self.config.signal.use_relaxed_volume,
                min_confirmation_bars=self.config.signal.min_confirmation_bars,
                sustained_bars_required=self.config.signal.sustained_bars_required,
                signal_cooldown_bars=self.config.signal.signal_cooldown_bars,
                use_or_bias_filter=self.config.signal.use_or_bias_filter,
                use_time_filter=self.config.time.use_time_filter,
                rth_only=self.config.time.rth_only
            )
            
            # Initialize position manager if client available
            if self.client:
                self.position_manager = PositionManager(
                    client=self.client,
                    symbol=self.config.trading.symbol,
                    contracts=self.config.trading.contracts,
                    max_daily_trades=self.config.trading.max_daily_trades,
                    min_dte=self.config.trading.min_days_to_expiry,
                    max_dte=self.config.trading.max_days_to_expiry,
                    paper_trading=self.config.paper_trading,
                    enable_stop_loss=self.config.trading.enable_stop_loss,
                    stop_loss_percent=self.config.trading.stop_loss_percent,
                    stop_loss_dollars=self.config.trading.stop_loss_dollars,
                    enable_take_profit=self.config.trading.enable_take_profit,
                    take_profit_percent=self.config.trading.take_profit_percent,
                    take_profit_dollars=self.config.trading.take_profit_dollars,
                    enable_trailing_stop=self.config.trading.enable_trailing_stop,
                    trailing_stop_percent=self.config.trading.trailing_stop_percent,
                    trailing_stop_activation=self.config.trading.trailing_stop_activation
                )
                
        except Exception as e:
            print(f"Initialization error: {e}")
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected WebSocket clients"""
        for ws in self.connected_websockets:
            try:
                await ws.send_json(message)
            except:
                pass

# Global state instance
state = BotState()

# ==================== API ROUTES ====================

@app.get("/api/status")
async def get_status():
    """Get current bot status"""
    detector_state = state.detector.get_state_summary() if state.detector else {}
    position_summary = state.position_manager.get_position_summary() if state.position_manager else {"status": "FLAT"}
    daily_stats = state.position_manager.get_daily_stats() if state.position_manager else {}
    
    return {
        "connected": state.client is not None,
        "authenticated": state.auth.is_authenticated if state.auth else False,
        "running": state.is_running,
        "paper_trading": state.config.paper_trading,
        "current_price": state.current_price,
        "last_update": state.last_update.isoformat() if state.last_update else None,
        "position": position_summary,
        "daily_stats": daily_stats,
        "detector": detector_state,
        "bars_count": len(state.bars),
        "signals_count": len(state.signals)
    }


@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "trading": {
            "symbol": state.config.trading.symbol,
            "contracts": state.config.trading.contracts,
            "max_daily_trades": state.config.trading.max_daily_trades,
            "min_days_to_expiry": state.config.trading.min_days_to_expiry,
            "max_days_to_expiry": state.config.trading.max_days_to_expiry,
            "enable_stop_loss": state.config.trading.enable_stop_loss,
            "stop_loss_percent": state.config.trading.stop_loss_percent,
            "stop_loss_dollars": state.config.trading.stop_loss_dollars,
            "enable_take_profit": state.config.trading.enable_take_profit,
            "take_profit_percent": state.config.trading.take_profit_percent,
            "take_profit_dollars": state.config.trading.take_profit_dollars,
            "enable_trailing_stop": state.config.trading.enable_trailing_stop,
            "trailing_stop_percent": state.config.trading.trailing_stop_percent,
            "trailing_stop_activation": state.config.trading.trailing_stop_activation,
        },
        "signal": {
            "use_or_bias_filter": state.config.signal.use_or_bias_filter,
            "signal_cooldown_bars": state.config.signal.signal_cooldown_bars,
            "length_period": state.config.signal.length_period,
            "volume_threshold": state.config.signal.volume_threshold,
        },
        "time": {
            "rth_only": state.config.time.rth_only,
            "use_time_filter": state.config.time.use_time_filter,
        },
        "paper_trading": state.config.paper_trading
    }


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    """Update configuration"""
    if update.paper_trading is not None:
        state.config.paper_trading = update.paper_trading
    if update.contracts is not None:
        state.config.trading.contracts = update.contracts
    if update.max_daily_trades is not None:
        state.config.trading.max_daily_trades = update.max_daily_trades
    if update.max_days_to_expiry is not None:
        state.config.trading.max_days_to_expiry = update.max_days_to_expiry
    if update.enable_stop_loss is not None:
        state.config.trading.enable_stop_loss = update.enable_stop_loss
    if update.stop_loss_percent is not None:
        state.config.trading.stop_loss_percent = update.stop_loss_percent
    if update.stop_loss_dollars is not None:
        state.config.trading.stop_loss_dollars = update.stop_loss_dollars
    if update.enable_take_profit is not None:
        state.config.trading.enable_take_profit = update.enable_take_profit
    if update.take_profit_percent is not None:
        state.config.trading.take_profit_percent = update.take_profit_percent
    if update.take_profit_dollars is not None:
        state.config.trading.take_profit_dollars = update.take_profit_dollars
    if update.enable_trailing_stop is not None:
        state.config.trading.enable_trailing_stop = update.enable_trailing_stop
    if update.trailing_stop_percent is not None:
        state.config.trading.trailing_stop_percent = update.trailing_stop_percent
    if update.trailing_stop_activation is not None:
        state.config.trading.trailing_stop_activation = update.trailing_stop_activation
    if update.use_or_bias_filter is not None:
        state.config.signal.use_or_bias_filter = update.use_or_bias_filter
    if update.rth_only is not None:
        state.config.time.rth_only = update.rth_only
    if update.signal_cooldown_bars is not None:
        state.config.signal.signal_cooldown_bars = update.signal_cooldown_bars
    
    # Reinitialize position manager with new settings
    if state.client and state.position_manager:
        state.position_manager = PositionManager(
            client=state.client,
            symbol=state.config.trading.symbol,
            contracts=state.config.trading.contracts,
            max_daily_trades=state.config.trading.max_daily_trades,
            min_dte=state.config.trading.min_days_to_expiry,
            max_dte=state.config.trading.max_days_to_expiry,
            paper_trading=state.config.paper_trading,
            enable_stop_loss=state.config.trading.enable_stop_loss,
            stop_loss_percent=state.config.trading.stop_loss_percent,
            stop_loss_dollars=state.config.trading.stop_loss_dollars,
            enable_take_profit=state.config.trading.enable_take_profit,
            take_profit_percent=state.config.trading.take_profit_percent,
            take_profit_dollars=state.config.trading.take_profit_dollars,
            enable_trailing_stop=state.config.trading.enable_trailing_stop,
            trailing_stop_percent=state.config.trading.trailing_stop_percent,
            trailing_stop_activation=state.config.trading.trailing_stop_activation
        )
    
    await state.broadcast({"type": "config_updated", "config": await get_config()})
    return {"status": "ok", "config": await get_config()}


@app.get("/api/quote")
async def get_quote():
    """Get current SPY quote"""
    if not state.client:
        raise HTTPException(status_code=503, detail="Not connected to Schwab")
    
    try:
        quote = state.client.get_quote(state.config.trading.symbol)
        state.current_price = quote.last_price
        state.last_update = datetime.now()
        
        return {
            "symbol": quote.symbol,
            "last": quote.last_price,
            "bid": quote.bid,
            "ask": quote.ask,
            "high": quote.high,
            "low": quote.low,
            "open": quote.open,
            "volume": quote.volume,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bars")
async def get_bars(limit: int = 100):
    """Get recent price bars"""
    return state.bars[-limit:]


@app.get("/api/signals")
async def get_signals(limit: int = 50):
    """Get recent signals"""
    return state.signals[-limit:]


@app.get("/api/levels")
async def get_levels():
    """Get current VAH/POC/VAL levels"""
    if not state.detector:
        return {"vah": 0, "poc": 0, "val": 0, "vwap": 0}
    
    levels = state.detector.get_current_levels()
    detector_state = state.detector.get_state_summary()
    
    return {
        "vah": levels.vah if levels else 0,
        "poc": levels.poc if levels else 0,
        "val": levels.val if levels else 0,
        "vwap": levels.vwap if levels else 0,
        "or_high": detector_state.get("or_high", 0),
        "or_low": detector_state.get("or_low", 0),
        "or_bias": detector_state.get("or_bias", "NEUTRAL"),
        "or_complete": detector_state.get("or_complete", False)
    }


@app.get("/api/position")
async def get_position():
    """Get current position"""
    if not state.position_manager:
        return {"status": "FLAT", "details": None}
    
    return state.position_manager.get_position_summary()


@app.get("/api/trades")
async def get_trades():
    """Get today's trades"""
    if not state.position_manager:
        return []
    
    today = date.today()
    trades = [
        {
            "id": t.id,
            "signal_type": t.signal_type.value,
            "direction": t.direction.value,
            "entry_time": t.entry_time.isoformat(),
            "exit_time": t.exit_time.isoformat() if t.exit_time else None,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "option_symbol": t.option_symbol,
            "pnl": t.pnl,
            "pnl_percent": t.pnl_percent,
            "exit_reason": t.exit_reason
        }
        for t in state.position_manager.trades
        if t.entry_time.date() == today
    ]
    return trades


@app.post("/api/signal/manual")
async def trigger_manual_signal(signal: ManualSignal):
    """Manually trigger a signal (for testing)"""
    if not state.position_manager:
        raise HTTPException(status_code=503, detail="Position manager not initialized")
    
    from signal_detector import Signal, SignalType
    
    direction = Direction.LONG if signal.direction.lower() == "long" else Direction.SHORT
    signal_type = SignalType.VAL_BOUNCE if direction == Direction.LONG else SignalType.VAH_REJECTION
    
    manual_signal = Signal(
        signal_type=signal_type,
        direction=direction,
        timestamp=datetime.now(),
        price=state.current_price,
        vah=0, val=0, poc=0,
        or_bias=0,
        reason=f"Manual {signal.direction} signal"
    )
    
    trade = state.position_manager.process_signal(manual_signal)
    
    if trade:
        signal_data = {
            "type": "signal",
            "signal_type": signal_type.value,
            "direction": direction.value,
            "price": state.current_price,
            "timestamp": datetime.now().isoformat(),
            "manual": True
        }
        state.signals.append(signal_data)
        await state.broadcast(signal_data)
        
        return {"status": "ok", "trade_id": trade.id}
    else:
        return {"status": "no_trade", "reason": "Trade not executed (locked out or already positioned)"}


@app.post("/api/position/close")
async def close_position():
    """Force close current position"""
    if not state.position_manager:
        raise HTTPException(status_code=503, detail="Position manager not initialized")
    
    closed = state.position_manager.force_close_all("Manual close from dashboard")
    
    await state.broadcast({"type": "position_closed", "trades": len(closed)})
    return {"status": "ok", "closed_trades": len(closed)}


@app.post("/api/bot/start")
async def start_bot():
    """Start the bot's main loop"""
    state.is_running = True
    await state.broadcast({"type": "bot_status", "running": True})
    return {"status": "ok", "running": True}


@app.post("/api/bot/stop")
async def stop_bot():
    """Stop the bot's main loop"""
    state.is_running = False
    await state.broadcast({"type": "bot_status", "running": False})
    return {"status": "ok", "running": False}


# ==================== WEBSOCKET ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    state.connected_websockets.append(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "connected",
            "status": await get_status()
        })
        
        # Keep connection alive and send updates
        while True:
            try:
                # Wait for message or timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                
                # Handle incoming messages
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    
            except asyncio.TimeoutError:
                # Send periodic updates
                if state.client and state.is_running:
                    try:
                        quote = state.client.get_quote(state.config.trading.symbol)
                        state.current_price = quote.last_price
                        state.last_update = datetime.now()
                        
                        await websocket.send_json({
                            "type": "price_update",
                            "price": quote.last_price,
                            "bid": quote.bid,
                            "ask": quote.ask,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Also send levels
                        levels = await get_levels()
                        await websocket.send_json({
                            "type": "levels_update",
                            **levels
                        })
                        
                    except Exception as e:
                        print(f"Error fetching quote: {e}")
                        
    except WebSocketDisconnect:
        state.connected_websockets.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in state.connected_websockets:
            state.connected_websockets.remove(websocket)


# ==================== MAIN ====================

# Serve frontend
@app.get("/")
async def serve_frontend():
    """Serve the dashboard frontend"""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"error": "Frontend not found"}


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         ToS Trading Bot - Dashboard Server                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    print("Starting server at http://localhost:8000")
    print("API docs at http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
