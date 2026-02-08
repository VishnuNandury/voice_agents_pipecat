#
# Multi-Pipeline Server for Loan Collection Voice Agent
#
# Spawns separate bot processes with different STT+TTS configurations.
# Each "Connect" button in the UI launches a bot with a specific pipeline.
#
# Usage:
#   python server.py
#

import asyncio
import os
import subprocess
import sys
import time

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

REQUIRED_KEYS = ["DEEPGRAM_API_KEY", "OPENAI_API_KEY"]
missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing:
    logger.error(f"Missing required environment variables: {', '.join(missing)}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Loan Collection Voice Agent - Multi Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track running bot processes: port -> (subprocess, log_file_handle)
bot_processes: dict[int, tuple[subprocess.Popen, object]] = {}
BASE_BOT_PORT = 7870  # Bot instances run on 7870, 7871, 7872, 7873

# Log directory for bot outputs
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def get_port_for_config(stt: str, tts: str) -> int:
    """Deterministic port for each STT+TTS combination."""
    combos = {
        ("deepgram", "openai"): 0,
        ("deepgram", "edge"): 1,
        ("whisper", "openai"): 2,
        ("whisper", "edge"): 3,
    }
    return BASE_BOT_PORT + combos.get((stt, tts), 0)


def stop_bot(port: int):
    """Stop a running bot process on a given port."""
    entry = bot_processes.get(port)
    if entry:
        proc, log_fh = entry
        if proc.poll() is None:
            logger.info(f"Stopping bot on port {port}")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if log_fh:
            try:
                log_fh.close()
            except Exception:
                pass
    bot_processes.pop(port, None)


def start_bot(stt: str, tts: str) -> int:
    """Start a bot process with the given STT+TTS config. Returns the port."""
    port = get_port_for_config(stt, tts)

    # Stop existing bot on this port
    stop_bot(port)

    env = os.environ.copy()
    env["PIPELINE_STT"] = stt
    env["PIPELINE_TTS"] = tts
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        sys.executable, "bot.py",
        "-t", "webrtc",
        "--port", str(port),
    ]

    # Redirect stdout/stderr to a log file so the pipe doesn't fill up and block
    log_path = os.path.join(LOG_DIR, f"bot_{stt}_{tts}_{port}.log")
    log_fh = open(log_path, "w", encoding="utf-8")

    logger.info(f"Starting bot: STT={stt} TTS={tts} on port {port}")
    logger.info(f"Bot logs: {log_path}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    bot_processes[port] = (proc, log_fh)
    return port


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the multi-pipeline UI."""
    with open(
        os.path.join(os.path.dirname(__file__), "static", "index.html"),
        "r",
        encoding="utf-8",
    ) as f:
        return HTMLResponse(content=f.read())


@app.post("/launch")
async def launch_bot(request: Request):
    """Launch a bot with a specific STT+TTS configuration."""
    body = await request.json()
    stt = body.get("stt", "deepgram")
    tts = body.get("tts", "openai")

    if stt not in ("deepgram", "whisper"):
        return JSONResponse({"error": "Invalid STT"}, status_code=400)
    if tts not in ("openai", "edge"):
        return JSONResponse({"error": "Invalid TTS"}, status_code=400)

    port = start_bot(stt, tts)

    # Give the bot a moment to start
    await asyncio.sleep(4)

    # Check if process is still running
    entry = bot_processes.get(port)
    if entry:
        proc, log_fh = entry
        if proc.poll() is not None:
            # Process died, read log for error
            log_path = os.path.join(LOG_DIR, f"bot_{stt}_{tts}_{port}.log")
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    output = f.read()[-800:]
            except Exception:
                output = "Could not read log file"
            logger.error(f"Bot failed to start on port {port}: {output}")
            return JSONResponse(
                {"error": "Bot failed to start", "details": output},
                status_code=500,
            )

    return JSONResponse({
        "status": "running",
        "port": port,
        "stt": stt,
        "tts": tts,
        "client_url": f"http://localhost:{port}/client",
    })


@app.post("/stop")
async def stop_bot_endpoint(request: Request):
    """Stop a running bot."""
    body = await request.json()
    port = body.get("port")
    if port and port in bot_processes:
        stop_bot(port)
        return JSONResponse({"status": "stopped"})
    return JSONResponse({"status": "not_running"})


@app.get("/logs/{port}")
async def get_bot_logs(port: int):
    """Fetch the last N lines of a bot's log file."""
    for stt in ("deepgram", "whisper"):
        for tts in ("openai", "edge"):
            log_path = os.path.join(LOG_DIR, f"bot_{stt}_{tts}_{port}.log")
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    return JSONResponse({
                        "port": port,
                        "lines": len(lines),
                        "tail": "".join(lines[-50:]),
                    })
                except Exception as e:
                    return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"error": "Log file not found"}, status_code=404)


@app.get("/health")
async def health():
    running = {}
    for port, entry in bot_processes.items():
        proc, _ = entry
        running[port] = proc.poll() is None
    return {"status": "ok", "bots": running}


@app.on_event("shutdown")
async def shutdown():
    """Clean up all bot processes on server shutdown."""
    for port in list(bot_processes.keys()):
        stop_bot(port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SERVER_PORT", "7860"))
    logger.info(f"Starting Multi-Pipeline Server on port {port}")
    logger.info(f"Open http://localhost:{port} in your browser")
    logger.info(f"Bot logs directory: {LOG_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=port)
