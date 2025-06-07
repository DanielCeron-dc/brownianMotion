import time
import socketio
import numpy as np
from aiohttp import web
import asyncio
import os

# === GLOBAL STATE SHARED BETWEEN THREADS ===

paths = 5
dt = 0.2              # seconds per simulation step
mu_log = 0.10         # log‐drift (e.g. 10% annualized)
sigma_log = 0.30      # log‐volatility (e.g. 30% annualized)
rng = np.random.default_rng(4)

# Use Geometric Brownian Motion so prices stay positive.
current_price = np.full(paths, 100.0)  # start all at $100

# Precompute the "drift adjustment" term:
drift_adj = (mu_log - 0.5 * sigma_log**2) * dt

# Build candles at a slower interval:
# For example, one candle per 1 second → CANDLE_INTERVAL = 1.0
# CANDLE_STEPS = number of dt‐steps per candle
CANDLE_INTERVAL = 1.0
CANDLE_STEPS = int(CANDLE_INTERVAL / dt)  # e.g. 1.0/0.2 = 5 steps per candle

# === STOCK DATA (one‐to‐one with GBM paths) ===

HISTORY_LIMIT = 30
INITIAL_STOCKS = [
    {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "price": 100.0,
        "daily_change": 0.0,
        "candles": [],  # list of {open, high, low, close}
    },
    {
        "symbol": "GOOGL",
        "name": "Alphabet Inc.",
        "price": 100.0,
        "daily_change": 0.0,
        "candles": [],
    },
    {
        "symbol": "MSFT",
        "name": "Microsoft Corp.",
        "price": 100.0,
        "daily_change": 0.0,
        "candles": [],
    },
    {
        "symbol": "AMZN",
        "name": "Amazon.com, Inc.",
        "price": 100.0,
        "daily_change": 0.0,
        "candles": [],
    },
    {
        "symbol": "TSLA",
        "name": "Tesla, Inc.",
        "price": 100.0,
        "daily_change": 0.0,
        "candles": [],
    },
]
STOCKS = {stock["symbol"]: stock for stock in INITIAL_STOCKS}
stock_symbols = list(STOCKS.keys())  # ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

# Track the "open price" for each candle separately
last_candle_open = current_price.copy()

# === SOCKET.IO + AIOHTTP SETUP ===

sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

@sio.event
async def connect(sid, environ):
    print(f"[Socket.IO] Client connected: {sid}")
    # 1) Send initial stock list (all prices = 100.0, empty candles)
    await sio.emit("onInitialStocks", list(STOCKS.values()), to=sid)
    # 2) Send initial price snapshot
    await sio.emit("brownian_update", {"values": current_price.tolist()}, to=sid)

@sio.event
async def disconnect(sid):
    print(f"[Socket.IO] Client disconnected: {sid}")

# Add a simple health check endpoint
async def health_check(request):
    return web.Response(text="OK")

# Add a simple frontend for testing
async def index(request):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Brownian Motion Stock Simulator</title>
        <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    </head>
    <body>
        <h1>Brownian Motion Stock Simulator</h1>
        <div id="stocks"></div>
        <div id="prices"></div>
        
        <script>
            const socket = io();
            
            socket.on('connect', function() {
                console.log('Connected to server');
            });
            
            socket.on('onInitialStocks', function(stocks) {
                console.log('Initial stocks:', stocks);
                const stocksDiv = document.getElementById('stocks');
                stocksDiv.innerHTML = '<h2>Stocks:</h2>' + 
                    stocks.map(stock => `<p>${stock.symbol}: ${stock.name} - $${stock.price.toFixed(2)}</p>`).join('');
            });
            
            socket.on('brownian_update', function(data) {
                const pricesDiv = document.getElementById('prices');
                pricesDiv.innerHTML = '<h2>Current Prices:</h2>' + 
                    data.values.map((price, i) => `<p>Path ${i}: $${price.toFixed(2)}</p>`).join('');
            });
            
            socket.on('onChangeStock', function(stock) {
                console.log('Stock update:', stock);
            });
        </script>
    </body>
    </html>
    """
    return web.Response(text=html, content_type='text/html')

async def simulation_loop():
    """
    Indefinitely:
      1) Draw fresh Z_i ~ N(0,1) for each path i=0..4
      2) Update price_i ← price_i * exp(drift_adj + sigma_log*sqrt(dt)*Z_i)
      3) Emit "brownian_update" with the updated prices
      4) For each i, compute daily_change vs. previous price immediately and emit "onChangeStock"
         (with price and daily_change but candles only updated every CANDLE_STEPS)
      5) Every CANDLE_STEPS iterations, build one new candle per stock:
         - open = last_candle_open[i]
         - high = max over that interval (we'll track a running high/low)
         - low = min over that interval
         - close = current_price[i]
         Append that candle to stock["candles"], capped at HISTORY_LIMIT
         Then reset last_candle_open[i] = current_price[i] and reset high/low trackers.
      6) Sleep dt seconds and repeat.
    """
    prev_prices = np.copy(current_price)
    step_counter = 0

    # Track high/low within current candle interval for each path
    candle_high = current_price.copy()
    candle_low = current_price.copy()

    while True:
        ts = int(time.time())

        # 1) Draw new normals
        Z = rng.normal(0.0, 1.0, size=paths)
        shock = sigma_log * np.sqrt(dt) * Z

        # 2) Geometric Brownian update
        current_price[:] = current_price * np.exp(drift_adj + shock)

        # Update high/low trackers
        candle_high[:] = np.maximum(candle_high, current_price)
        candle_low[:] = np.minimum(candle_low, current_price)

        # 3) Broadcast updated prices
        await sio.emit("brownian_update", {"values": current_price.tolist()})

        # 4) Immediately update stocks' price and daily_change
        for i, symbol in enumerate(stock_symbols):
            stock = STOCKS[symbol]
            old_price = prev_prices[i]
            new_price = current_price[i]

            if old_price == 0.0:
                daily_change = 0.0
            else:
                daily_change = round(((new_price - old_price) / abs(old_price)) * 100, 2)

            stock["price"] = float(new_price)
            stock["daily_change"] = float(daily_change)

            # We do NOT append a candle here; wait until CANDLE_STEPS steps.

            await sio.emit("onChangeStock", {
                "symbol": stock["symbol"],
                "name": stock["name"],
                "price": stock["price"],
                "daily_change": stock["daily_change"],
                "candles": stock["candles"]
            })

        # 5) Build a candle every CANDLE_STEPS iterations
        step_counter += 1
        if step_counter >= CANDLE_STEPS:
            for i, symbol in enumerate(stock_symbols):
                stock = STOCKS[symbol]
                open_price = float(last_candle_open[i])
                high_price = float(candle_high[i])
                low_price = float(candle_low[i])
                close_price = float(current_price[i])

                candle = {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price
                }
                stock["candles"].append(candle)
                if len(stock["candles"]) > HISTORY_LIMIT:
                    stock["candles"].pop(0)

                # After building the candle, reset trackers:
                last_candle_open[i] = current_price[i]
                candle_high[i] = current_price[i]
                candle_low[i] = current_price[i]

                # Emit updated stock including its updated candles
                await sio.emit("onChangeStock", {
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "price": stock["price"],
                    "daily_change": stock["daily_change"],
                    "candles": stock["candles"]
                })

            step_counter = 0  # reset for next candle interval

        # 6) Save for next iteration's percent change
        prev_prices[:] = current_price[:]

        # 7) Sleep dt seconds
        await asyncio.sleep(dt)

async def on_startup(app):
    app["task"] = asyncio.create_task(simulation_loop())

async def on_cleanup(app):
    app["task"].cancel()
    try:
        await app["task"]
    except asyncio.CancelledError:
        pass

app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)

# Add routes
app.router.add_get('/', index)
app.router.add_get('/health', health_check)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    web.run_app(app, port=port, host='0.0.0.0')

