import threading
import time
import socketio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from aiohttp import web
import asyncio  # for asyncio.sleep in simulation_loop

# === GLOBAL STATE SHARED BETWEEN THREADS ===

paths = 5
dt = 0.2            # seconds per simulation step
mu_log = 0.10       # log‐drift (e.g. 10% annualized)
sigma_log = 0.30    # log‐volatility (e.g. 30% annualized)
rng = np.random.default_rng(4)

# Use Geometric Brownian Motion so prices stay positive.
current_price = np.full(paths, 100.0)  # start all at $100

# Precompute the “drift adjustment” term:
drift_adj = (mu_log - 0.5 * sigma_log**2) * dt

# === STOCK DATA (one‐to‐one with GBM paths) ===

HISTORY_LIMIT = 30
INITIAL_STOCKS = [
    {"symbol": "AAPL",  "name": "Apple Inc.",       "price": 100.0, "daily_change": 0.0, "candles": [], "history": []},
    {"symbol": "GOOGL", "name": "Alphabet Inc.",    "price": 100.0, "daily_change": 0.0, "candles": [], "history": []},
    {"symbol": "MSFT",  "name": "Microsoft Corp.",  "price": 100.0, "daily_change": 0.0, "candles": [], "history": []},
    {"symbol": "AMZN",  "name": "Amazon.com, Inc.", "price": 100.0, "daily_change": 0.0, "candles": [], "history": []},
    {"symbol": "TSLA",  "name": "Tesla, Inc.",      "price": 100.0, "daily_change": 0.0, "candles": [], "history": []},
]
STOCKS = {stock["symbol"]: stock for stock in INITIAL_STOCKS}
stock_symbols = list(STOCKS.keys())  # ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

# === SOCKET.IO + AIOHTTP SETUP ===

sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

@sio.event
async def connect(sid, environ):
    print(f"[Socket.IO] Client connected: {sid}")
    # 1) Send initial stock list (all prices = 100.0 initially)
    await sio.emit("onInitialStocks", list(STOCKS.values()), to=sid)
    # 2) Send initial price snapshot
    await sio.emit("brownian_update", {"values": current_price.tolist()}, to=sid)

@sio.event
async def disconnect(sid):
    print(f"[Socket.IO] Client disconnected: {sid}")

async def simulation_loop():
    """
    Indefinitely:
      1) Draw fresh Z_i ~ N(0,1) for each path i=0..4
      2) Update price_i ← price_i * exp(drift_adj + sigma_log*sqrt(dt)*Z_i)
      3) Emit "brownian_update" with the updated prices
      4) For each i, compute daily_change vs. previous price, append history, and emit "onChangeStock".
      5) Sleep dt seconds and repeat.
    """
    prev_prices = np.copy(current_price)
    while True:
        ts = int(time.time())

        # 1) Draw new normals
        Z = rng.normal(0.0, 1.0, size=paths)
        shock = sigma_log * np.sqrt(dt) * Z

        # 2) Geometric Brownian update
        current_price[:] = current_price * np.exp(drift_adj + shock)

        # DEBUG: print every 50 steps
        print("DEBUG Prices:", np.round(current_price, 2))

        # 3) Broadcast updated prices
        await sio.emit("brownian_update", {"values": current_price.tolist()})

        # 4) Update each stock from its GBM path
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

            entry = {
                "timestamp": ts,
                "price": float(new_price),
                "daily_change": float(daily_change),
            }
            stock["history"].append(entry)
            if len(stock["history"]) > HISTORY_LIMIT:
                stock["history"].pop(0)

            await sio.emit("onChangeStock", stock)

        # 5) Save for next iteration
        prev_prices[:] = current_price[:]

        # 6) Sleep dt seconds
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

def run_server():
    # Run aiohttp server in background thread without signal handlers
    web.run_app(app, port=5004, handle_signals=False)

# === LIVE PLOT WITH SEABORN AND FUNCANIMATION (MAIN THREAD) ===

def run_live_plot():
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Live GBM Prices (μ_log=0.10, σ_log=0.30; running indefinitely)")
    ax.set_xlabel("Time Steps (rolling)")
    ax.set_ylabel("Price")

    # Create five Line2D objects, labeled with stock symbols
    lines = []
    for symbol in stock_symbols:
        (line,) = ax.plot([], [], label=symbol)
        lines.append(line)
    ax.legend(loc="upper left")

    BUFFER_LEN = 100
    data_buffer = np.zeros((paths, BUFFER_LEN))
    time_buffer = np.arange(-BUFFER_LEN + 1, 1)

    def init():
        for ln in lines:
            ln.set_data([], [])
        return lines

    def update(frame):
        # Shift buffer left
        data_buffer[:, :-1] = data_buffer[:, 1:]
        time_buffer[:-1] = time_buffer[1:]

        # Insert newest prices at end
        data_buffer[:, -1] = current_price[:]
        time_buffer[-1] = time_buffer[-2] + 1 if BUFFER_LEN > 1 else 0

        # Update each line’s data
        for i, ln in enumerate(lines):
            ln.set_data(time_buffer, data_buffer[i])

        # Rescale Y-axis to current range
        flat = data_buffer.flatten()
        y_min, y_max = flat.min(), flat.max()
        if y_max == y_min:
            ax.set_ylim(y_min - 1.0, y_max + 1.0)
        else:
            margin = (y_max - y_min) * 0.15
            ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlim(time_buffer[0], time_buffer[-1])

        return lines

    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        interval=dt * 1000,  # dt seconds → milliseconds
        blit=False,
        cache_frame_data=False,
    )

    plt.show()  # blocks until window is closed

if __name__ == "__main__":
    # 1) Start the Socket.IO + aiohttp server in a daemon thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("🔌 Server is running on http://localhost:5004 …")

    # 2) Run the Seaborn live plot in the MAIN thread
    run_live_plot()

    print("Plot closed; exiting.")