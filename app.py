#
# Cloud-Ready Server for Loan Collection Voice Agent
#
# Custom FastAPI server with TURN/ICE server support for cloud deployments
# (Render, AWS, etc.) where NAT traversal is required.
#
# Usage:
#   python app.py                          # Local
#   gunicorn app:app -k uvicorn.workers.UvicornWorker  # Production
#

import asyncio
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from loguru import logger

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Validate required keys
# ---------------------------------------------------------------------------

REQUIRED_KEYS = ["DEEPGRAM_API_KEY", "OPENAI_API_KEY"]
missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing:
    logger.error(f"Missing environment variables: {', '.join(missing)}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# ICE / TURN configuration
# ---------------------------------------------------------------------------

from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)


def build_ice_servers() -> List[IceServer]:
    """Build ICE server list from environment variables."""
    servers = []

    # Always include a STUN server
    servers.append(IceServer(urls="stun:stun.l.google.com:19302"))

    # Add TURN server if credentials provided
    turn_url = os.getenv("TURN_URL")
    turn_username = os.getenv("TURN_USERNAME")
    turn_credential = os.getenv("TURN_CREDENTIAL")

    if turn_url and turn_username and turn_credential:
        logger.info(f"TURN server configured: {turn_url}")
        servers.append(
            IceServer(
                urls=turn_url,
                username=turn_username,
                credential=turn_credential,
            )
        )
        # Also add TURNS (TLS) variant if it's a turn: URL
        if turn_url.startswith("turn:"):
            turns_url = turn_url.replace("turn:", "turns:", 1)
            # Add port 443 for TURNS if no port specified
            if ":" not in turns_url.split("turns:")[1].split("?")[0].rsplit(":", 1)[-1]:
                pass  # Keep as-is, server handles port
            servers.append(
                IceServer(
                    urls=turns_url,
                    username=turn_username,
                    credential=turn_credential,
                )
            )
    else:
        logger.warning(
            "No TURN server configured. WebRTC may fail behind NAT. "
            "Set TURN_URL, TURN_USERNAME, TURN_CREDENTIAL in .env"
        )

    return servers


# Build ICE config once at startup
ICE_SERVERS = build_ice_servers()

# Client-side ICE config (returned in /start response)
ICE_CONFIG_FOR_CLIENT = {
    "iceServers": [
        {"urls": s.urls, **({"username": s.username} if s.username else {}),
         **({"credential": s.credential} if s.credential else {})}
        for s in ICE_SERVERS
    ]
}

# ---------------------------------------------------------------------------
# Import bot module
# ---------------------------------------------------------------------------

# Force single-pipeline mode for cloud (one bot per deployment)
stt_type = os.getenv("PIPELINE_STT", "deepgram")
tts_type = os.getenv("PIPELINE_TTS", "openai")
logger.info(f"Pipeline: STT={stt_type} | TTS={tts_type}")

# Import bot module
import bot as bot_module

# ---------------------------------------------------------------------------
# SmallWebRTC handler with TURN support
# ---------------------------------------------------------------------------

small_webrtc_handler = SmallWebRTCRequestHandler(ice_servers=ICE_SERVERS)

# In-memory session store
active_sessions: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await small_webrtc_handler.close()


app = FastAPI(title="Loan Collection Voice Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the prebuilt WebRTC client UI
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

app.mount("/client", SmallWebRTCPrebuiltUI)


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline": {"stt": stt_type, "tts": tts_type},
        "turn_configured": any(
            s.username for s in ICE_SERVERS if hasattr(s, "username") and s.username
        ),
    }


# ---------------------------------------------------------------------------
# WebRTC signaling routes (replicating PipeCat runner with TURN support)
# ---------------------------------------------------------------------------

@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """Handle WebRTC offer from the prebuilt client."""
    from pipecat.runner.types import SmallWebRTCRunnerArguments

    async def webrtc_connection_callback(connection: SmallWebRTCConnection):
        runner_args = SmallWebRTCRunnerArguments(
            webrtc_connection=connection,
            body=request.request_data,
        )
        background_tasks.add_task(bot_module.bot, runner_args)

    answer = await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=webrtc_connection_callback,
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    """Handle trickle ICE candidates."""
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


@app.post("/start")
async def rtvi_start(request: Request):
    """Return session ID and ICE config (including TURN) to the client."""
    try:
        request_data = await request.json()
    except Exception:
        request_data = {}

    session_id = str(uuid.uuid4())
    active_sessions[session_id] = request_data.get("body", {})

    # Always return ICE config with TURN servers
    return {
        "sessionId": session_id,
        "iceConfig": ICE_CONFIG_FOR_CLIENT,
    }


@app.api_route(
    "/sessions/{session_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_request(
    session_id: str, path: str, request: Request, background_tasks: BackgroundTasks
):
    """Handle session-scoped requests (PipeCat Cloud compatibility)."""
    active_session = active_sessions.get(session_id)
    if active_session is None:
        return Response(content="Invalid session", status_code=404)

    if path.endswith("api/offer"):
        try:
            request_data = await request.json()
            if request.method == "POST":
                webrtc_request = SmallWebRTCRequest(
                    sdp=request_data["sdp"],
                    type=request_data["type"],
                    pc_id=request_data.get("pc_id"),
                    restart_pc=request_data.get("restart_pc"),
                    request_data=request_data.get("request_data")
                    or request_data.get("requestData")
                    or active_session,
                )
                return await offer(webrtc_request, background_tasks)
            elif request.method == "PATCH":
                patch_request = SmallWebRTCPatchRequest(
                    pc_id=request_data["pc_id"],
                    candidates=[
                        IceCandidate(**c) for c in request_data.get("candidates", [])
                    ],
                )
                return await ice_candidate(patch_request)
        except Exception as e:
            logger.error(f"WebRTC proxy error: {e}")
            return Response(content="Invalid request", status_code=400)

    return Response(status_code=200)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", os.getenv("SERVER_PORT", "7860")))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting on {host}:{port}")
    logger.info(f"Pipeline: STT={stt_type} | TTS={tts_type}")
    logger.info(f"ICE servers: {len(ICE_SERVERS)} configured")
    uvicorn.run(app, host=host, port=port)
