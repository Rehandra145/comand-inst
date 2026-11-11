"""
FastAPI Server & WebSocket Bridge — Web Interface untuk Robot Voice Command.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from hw_monitor import hw_monitor
from audio import list_audio_devices, stt as _stt, tts as _tts
from audio.tts import speak, is_tts_available
from web.listener import get_coordinator, WebVoiceCoordinator

logger = logging.getLogger("web.server")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Robot Voice Command & NLU Interface",
    description="Web Interface & REST/WebSocket API untuk Voice Command Robot Layanan RS",
    version="1.0.0",
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PYDANTIC SCHEMAS
# =========================

class TextInput(BaseModel):
    text: str

class MicConfig(BaseModel):
    duration: Optional[int] = 8
    device: Optional[int] = None

class Ros2ConfigInput(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = None
    mock: Optional[bool] = None

class TTSInput(BaseModel):
    text: str


# =========================
# WEBSOCKET CONNECTION MANAGER
# =========================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"[WebSocket] Client terhubung. Total klien: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"[WebSocket] Client terputus. Sisa klien: {len(self.active_connections)}")

    def broadcast_sync(self, payload: Dict[str, Any]):
        """Kirim pesan sinkron dari thread luar ke event loop FastAPI."""
        if not self.active_connections or self.loop is None:
            return
        
        asyncio.run_coroutine_threadsafe(self.broadcast(payload), self.loop)

    async def broadcast(self, payload: Dict[str, Any]):
        if not self.active_connections:
            return
        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_json(payload)
            except Exception:
                to_remove.append(connection)
        for conn in to_remove:
            self.disconnect(conn)


manager = ConnectionManager()


# =========================
# LIFECYCLE & BACKGROUND TASKS
# =========================

telemetry_task = None

async def telemetry_worker():
    """Mengirim data telemetri hardware Jetson secara periodik ke Web clients."""
    while True:
        try:
            if manager.active_connections:
                snap = hw_monitor.get_snapshot()
                coord = get_coordinator()
                await manager.broadcast({
                    "type": "telemetry",
                    "data": snap,
                    "state": coord.state,
                })
        except Exception as e:
            logger.debug(f"[TelemetryWorker] Error: {e}")
        await asyncio.sleep(1.0)


@app.on_event("startup")
async def startup_event():
    manager.loop = asyncio.get_running_loop()
    coord = get_coordinator()
    coord.add_subscriber(manager.broadcast_sync)
    
    # Jalankan background telemetry broadcaster
    global telemetry_task
    telemetry_task = asyncio.create_task(telemetry_worker())

    # Preload model STT di background agar tekan mic pertama tidak menunggu ~11 detik
    def _preload():
        try:
            _stt.get_recognizer()
            logger.info("[Web] Model STT siap.")
        except Exception as e:
            logger.warning(f"[Web] Gagal preload STT: {e}")
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _preload)
    logger.info("[Web] Server dan background broadcaster berhasil diinisialisasi.")


@app.on_event("shutdown")
async def shutdown_event():
    global telemetry_task
    if telemetry_task:
        telemetry_task.cancel()
    coord = get_coordinator()
    coord.remove_subscriber(manager.broadcast_sync)


# =========================
# REST API ENDPOINTS
# =========================

@app.get("/api/status")
async def get_system_status():
    """Mengambil status menyeluruh sistem, telemetri, model AI, dan konfigurasi ROS2."""
    coord = get_coordinator()
    snap = hw_monitor.get_snapshot()
    
    # Ambil status ROS 2
    ping_res = await asyncio.to_thread(coord.ping_ros2)
    ros_connected = ping_res.get("connected", False)
    ros_msg = ping_res.get("message", "")
    
    # Ambil list audio device mikrofon
    devices = []
    try:
        import sounddevice as sd
        raw_devs = sd.query_devices()
        for idx, d in enumerate(raw_devs):
            if d.get("max_input_channels", 0) > 0:
                devices.append({
                    "index": idx,
                    "name": d.get("name"),
                    "channels": d.get("max_input_channels"),
                    "sample_rate": d.get("default_samplerate"),
                })
    except Exception:
        pass

    return {
        "state": coord.state,
        "current_partial": coord.current_partial,
        "last_result": coord.last_result,
        "telemetry": snap,
        "ros2": {
            "mode": "mock" if coord.use_mock_ros2 else "socket",
            "host": coord.ros2_host or getattr(coord.ros2_adapter, "host", "10.107.245.221"),
            "port": coord.ros2_port or getattr(coord.ros2_adapter, "port", 5000),
            "connected": ros_connected,
            "message": ros_msg,
        },
        "devices": devices,
        "tts_enabled": coord.tts_enabled,
    }


@app.post("/api/mic/start")
async def start_mic(config: MicConfig = MicConfig()):
    """Memicu mikrofon fisik Jetson untuk mulai mendengarkan suara."""
    coord = get_coordinator()
    success = coord.start_listening(duration=config.duration, device=config.device)
    if not success:
        raise HTTPException(status_code=400, detail="Mikrofon sedang aktif mendengarkan/memproses.")
    return {"status": "ok", "message": "Mikrofon Jetson mulai mendengarkan"}


@app.post("/api/mic/stop")
async def stop_mic():
    """Menghentikan mikrofon fisik Jetson seketika."""
    coord = get_coordinator()
    coord.stop_listening()
    return {"status": "ok", "message": "Perintah stop dikirim ke mikrofon"}


@app.post("/api/nlu/process")
async def process_text_command(payload: TextInput):
    """Memproses teks perintah melalui NLU pipeline secara instan."""
    coord = get_coordinator()
    result = await asyncio.to_thread(coord.process_text, payload.text, "web_text")
    return {"status": "ok", "result": result}


@app.post("/api/ros2/config")
async def update_ros2_config(config: Ros2ConfigInput):
    """Memperbarui konfigurasi IP/Port ROS 2 dan mode simulasi/mock."""
    coord = get_coordinator()
    if config.mock is not None:
        coord.set_mock_mode(config.mock)
    if config.host and config.port:
        coord.update_ros2_target(config.host, config.port)
    return {"status": "ok", "message": "Konfigurasi ROS 2 diperbarui"}


@app.post("/api/ros2/ping")
async def ping_ros2_node():
    """Uji konektivitas TCP ke robot ROS 2."""
    coord = get_coordinator()
    return coord.ping_ros2()


@app.post("/api/tts/speak")
async def speak_text(payload: TTSInput):
    """Membunyikan teks lewat speaker Jetson menggunakan Piper TTS."""
    try:
        speak(payload.text, blocking=False)
        return {"status": "ok", "message": "Audio TTS diputar"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts/toggle")
async def toggle_tts():
    """Toggle apakah respon suara TTS robot aktif atau hening."""
    coord = get_coordinator()
    coord.tts_enabled = not coord.tts_enabled
    coord.broadcast({
        "type": "tts_toggle",
        "enabled": coord.tts_enabled
    })
    return {"status": "ok", "tts_enabled": coord.tts_enabled}


# =========================
# WEBSOCKET ENDPOINT
# =========================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    coord = get_coordinator()
    # Kirim initial handshake status
    await websocket.send_json({
        "type": "handshake",
        "state": coord.state,
        "mock": coord.use_mock_ros2,
        "last_result": coord.last_result,
    })

    loop = asyncio.get_running_loop()
    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # === Binary Frame: PCM audio dari browser mic ===
            if message.get("bytes"):
                coord.feed_browser_audio(message["bytes"])
                continue

            # === Text Frame: JSON command ===
            data = message.get("text", "")
            if not data:
                continue

            try:
                msg = json.loads(data)
                action = msg.get("action")

                if action == "start_mic":
                    duration = msg.get("duration", 8)
                    device = msg.get("device", None)
                    ok = coord.start_listening(duration=duration, device=device)
                    if not ok:
                        await websocket.send_json({"type": "busy", "state": coord.state})

                elif action == "stop_mic":
                    coord.stop_listening()

                elif action == "start_browser_mic":
                    success = coord.start_browser_mic_session(owner=websocket)
                    await websocket.send_json({
                        "type": "browser_mic_ready",
                        "success": success,
                        "state": coord.state,
                        "message": "OK" if success else "Sistem sedang sibuk",
                    })

                elif action == "stop_browser_mic":
                    coord.stop_browser_mic_session(cancel=bool(msg.get("cancel", False)))

                elif action == "client_log":
                    client = websocket.client.host if websocket.client else "?"
                    print(f"[CLIENT {client}] {msg.get('msg', '')}", flush=True)

                elif action == "reset":
                    coord.force_reset()

                elif action == "send_text":
                    text = msg.get("text", "")
                    if text:
                        # Jalankan di thread agar event loop tidak terblokir (NLU + TTS)
                        loop.run_in_executor(None, coord.process_text, text, "web_ws")

                elif action == "set_mock":
                    mock_enabled = bool(msg.get("mock", False))
                    await asyncio.to_thread(coord.set_mock_mode, mock_enabled)

                elif action == "ping_ros2":
                    res = await asyncio.to_thread(coord.ping_ros2)
                    await websocket.send_json({
                        "type": "ros2_ping_result",
                        **res
                    })

            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"[WebSocket] Exception: {e}")
    finally:
        manager.disconnect(websocket)
        # Jika HP terputus saat merekam, batalkan sesi agar state tidak macet
        if coord.owns_browser_session(websocket):
            coord.stop_browser_mic_session(cancel=True)


# =========================
# STATIC FRONTEND MOUNT
# =========================

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def run_web_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    ros2_host: Optional[str] = None,
    ros2_port: Optional[int] = None,
    ssl: bool = False,
):
    """Fungsi utama untuk menjalankan web server Uvicorn."""
    # Pastikan coordinator terinisialisasi dengan parameter host/port
    get_coordinator(ros2_host=ros2_host, ros2_port=ros2_port)

    # SSL support untuk browser mic dari HP/Laptop remote
    ssl_certfile = None
    ssl_keyfile = None
    scheme = "http"
    ws_scheme = "ws"

    if ssl:
        cert_dir = BASE_DIR / "certs"
        ssl_certfile = str(cert_dir / "cert.pem")
        ssl_keyfile = str(cert_dir / "key.pem")

        if not (cert_dir / "cert.pem").exists():
            import subprocess
            cert_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", ssl_keyfile, "-out", ssl_certfile,
                "-days", "3650", "-nodes",
                "-subj", "/CN=jetson-robot/O=RobotVoice/C=ID"
            ], capture_output=True)

        scheme = "https"
        ws_scheme = "wss"

    # Deteksi IP lokal untuk tampilkan instruksi yang benar
    local_ip = "0.0.0.0"
    try:
        import subprocess
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2)
        ips = result.stdout.strip().split()
        if ips:
            local_ip = ips[0]
    except Exception:
        pass

    print(f"\n" + "=" * 68)
    print(f"  🚀 ROBOT VOICE COMMAND — WEB DASHBOARD & API AKTIF")
    print(f"=" * 68)
    print(f"  ● Browser Lokal     : {scheme}://localhost:{port}")
    print(f"  ● Browser HP/Laptop : {scheme}://{local_ip}:{port}")
    print(f"  ● REST API Docs     : {scheme}://localhost:{port}/docs")
    print(f"  ● WebSocket         : {ws_scheme}://localhost:{port}/ws")
    if ssl:
        print(f"  ● SSL Mode          : AKTIF (Self-Signed Certificate)")
        print(f"  ● Browser Mic HP    : AKTIF (getUserMedia via HTTPS)")
    else:
        print(f"  ● Browser Mic HP    : Gunakan --ssl untuk mengaktifkan")
    print(f"=" * 68 + "\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )
