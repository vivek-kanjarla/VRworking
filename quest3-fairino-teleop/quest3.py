"""
quest3.py — reads 6-DOF controller pose from Meta Quest 3.

Three transport modes (set QUEST3_MODE in config.py):

  "oculus_reader"
      Uses the jborbik/oculus_reader APK (Quest 3 compatible fork).
      Setup:
        1. Sideload the APK:  adb install oculus_reader.apk
        2. Forward the port:  adb forward tcp:5555 tcp:5555
        3. Open the app on Quest 3 — it starts streaming immediately.
      Wire format: [uint32 length][pickled dict] over TCP.

  "vuer"
      Uses the Vuer WebXR library (pip install vuer).
      No APK needed — Quest 3 browser connects to a local HTTPS server.
      Setup:
        1. pip install vuer
        2. Generate SSL cert (see docs/quest3_setup.md)
        3. Point the Quest 3 browser at https://<your-pc-ip>:8012
        4. Accept the self-signed cert warning and grant controller access.
      Event: CONTROLLER_MOVE — 4×4 column-major SE(3) matrix per controller.

  "udp"
      Custom binary packet streamer. Packet format documented in _UDPTransport.

ControllerState is updated by a background worker (thread or process depending
on mode). If no new packet has arrived in QUEST3_STALE_MS ms, valid is set to
False so the teleop loop holds position rather than issuing a stale command.
"""

import multiprocessing
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from config import (
    QUEST3_MODE, QUEST3_ADB_PORT, QUEST3_UDP_PORT, QUEST3_ACTIVE_HAND,
    QUEST3_VUER_PORT, QUEST3_VUER_CERT, QUEST3_VUER_KEY,
)

QUEST3_STALE_MS = 1000  # ms before a cached state is treated as stale (vuer has ~60Hz gaps)

# Shared-array layout for Vuer process ↔ reader thread:
#   [0..2]  pos x, y, z   (metres)
#   [3..6]  quat qx,qy,qz,qw
#   [7]     trigger value  [0, 1]
#   [8]     squeeze value  [0, 1]
#   [9]     valid flag     (1.0 once first event received)
# Shared-array layout — right controller only:
#   [0..2]  pos x,y,z (m)
#   [3..6]  quat qx,qy,qz,qw
#   [7]     trigger
#   [8]     grip
#   [9]     valid flag
#   [10]    thumbstick X (-1=left, +1=right)
#   [11]    thumbstick Y (-1=up,   +1=down)
#   [12]    button A  (0/1)  — ACT record toggle   (right controller)
#   [13]    button B  (0/1)  — ACT success toggle  (right controller)
#   [14]    button X  (0/1)  — J6 roll CW          (left controller)
#   [15]    button Y  (0/1)  — J6 roll CCW         (left controller)
_VUER_SHARED_LEN = 16


# ── Quaternion / rotation helpers ────────────────────────────────────────────

@dataclass
class ControllerState:
    """Pose + input state for both Quest 3 controllers, in OpenXR convention.

    Right controller (primary — drives arm position):
      pos     — position in metres (x=right, y=up, z=back toward user)
      quat    — orientation quaternion [qx, qy, qz, qw]
      trigger — index trigger [0.0 → 1.0]
      grip    — grip button [0.0 → 1.0]
      valid   — False until first packet / if data gone stale

    Left controller (secondary — drives wrist J4/J5/J6 orientation):
      left_quat    — orientation quaternion [qx, qy, qz, qw]
      left_trigger — index trigger [0.0 → 1.0]
      left_valid   — False if left controller not tracked
    """
    pos:        np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat:       np.ndarray = field(default_factory=lambda: np.array([0., 0., 0., 1.]))
    trigger:    float = 0.0
    grip:       float = 0.0
    buttons:    dict  = field(default_factory=dict)
    valid:      bool  = False
    joystick_x: float = 0.0   # thumbstick X: -1=left,  +1=right  → J5
    joystick_y: float = 0.0   # thumbstick Y: -1=up,    +1=down   → J4


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion [qx, qy, qz, qw] (Shepperd method)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


# ── Vuer server process (top-level so multiprocessing can pickle it) ─────────

def _run_vuer_server(port: int, cert: str, key: str, hand: str,
                     shared: "multiprocessing.Array") -> None:  # noqa: F821
    """
    Runs inside a separate Process. Starts a Vuer HTTPS WebXR server, listens
    for CONTROLLER_MOVE events, and writes parsed pose into shared memory.

    shared layout: [px, py, pz, qx, qy, qz, qw, trigger, squeeze, valid_flag]
    """
    # HTTPS server for the WebXR HTML page (secure context → WebXR allowed).
    # Plain WebSocket server on port+1 for controller data.
    # ws://localhost is allowed from HTTPS pages as a loopback exception in Chrome.
    import asyncio, json, os, ssl, threading
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import websockets

    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webxr", "index.html")
    ws_port   = port + 1   # 8013 — plain WS, ADB reversed

    # ── HTTPS file server (serves index.html) ──────────────────────────────────
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                with open(html_path, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_error(500, str(e))
        def log_message(self, *_):
            pass   # silence access log

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    cert = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssl", "cert.pem")
    key  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssl", "key.pem")
    ssl_ctx.load_cert_chain(cert, key)
    https_srv = HTTPServer(("0.0.0.0", port), _Handler)
    https_srv.socket = ssl_ctx.wrap_socket(https_srv.socket, server_side=True)
    threading.Thread(target=https_srv.serve_forever, daemon=True).start()

    # ── Plain WebSocket server (controller data) ────────────────────────────────
    async def ws_handler(websocket):
        print("[QUEST3] Controller WebSocket connected — receiving pose data", flush=True)
        async for raw_msg in websocket:
            try:
                data    = json.loads(raw_msg)
                if "type" in data or "status" in data or "error" in data:
                    print(f"[QUEST3] Browser: {data}", flush=True)
                    if data.get("type") not in ("pose",):
                        continue  # don't try to parse non-pose messages as pose
                px, py, pz = float(data["x"]), float(data["y"]), float(data["z"])
                qx, qy, qz, qw = (float(data["qx"]), float(data["qy"]),
                                   float(data["qz"]), float(data["qw"]))
                trigger = float(data.get("trigger", 0.0))
                squeeze = float(data.get("grip",    0.0))
                joy_x   = float(data.get("joy_x",  0.0))
                joy_y   = float(data.get("joy_y",  0.0))
                btn_a   = float(data.get("btn_a",  0.0))
                btn_b   = float(data.get("btn_b",  0.0))
                btn_x   = float(data.get("btn_x",  0.0))
                btn_y   = float(data.get("btn_y",  0.0))
                with shared.get_lock():
                    shared[0] = px;  shared[1] = py;  shared[2] = pz
                    shared[3] = qx;  shared[4] = qy
                    shared[5] = qz;  shared[6] = qw
                    shared[7] = trigger; shared[8] = squeeze
                    shared[9] = 1.0
                    shared[10] = joy_x; shared[11] = joy_y
                    shared[12] = btn_a; shared[13] = btn_b
                    shared[14] = btn_x; shared[15] = btn_y
            except Exception:
                pass

    async def main():
        # WSS on ws_port — same SSL cert, WiFi connection (survives VR mode)
        ws_ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ws_ssl.load_cert_chain(cert, key)
        async with websockets.serve(ws_handler, "0.0.0.0", ws_port, ssl=ws_ssl):
            import socket as _socket
            try:
                s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                pc_ip = s.getsockname()[0]
                s.close()
            except Exception:
                pc_ip = "192.168.0.6"
            print(f"[QUEST3] Servers ready:\n"
                  f"  Page (HTTPS): https://{pc_ip}:{port}  ← open in Meta Browser\n"
                  f"  Data (WSS):   wss://{pc_ip}:{ws_port}  (WiFi, stays alive in VR)\n"
                  f"  Accept cert warning → tap anywhere → move right controller",
                  flush=True)
            await asyncio.Future()

    asyncio.run(main())


# ── Oculus Reader transport (ADB logcat) ──────────────────────────────────────

class _OculusReaderTransport:
    """
    Uses the jborbik/oculus_reader APK (Quest 3 compatible fork) via ADB logcat.

    Streams `adb logcat -s wE9ryARX:I` in a background thread. The APK logs one
    line per frame containing head-relative 4×4 SE(3) transforms and button state.

    Log line format (from Remotes.h):
      l:<16 floats>|r:<16 floats>&R,rightTrig <f>,rightGrip <f>,...
    'r' = right controller, 'l' = left (head-relative, row-major OVR::Matrix4f).
    Empty data ('&') means controllers not tracked — headset must be worn.

    No TCP port-forward needed. Requires USB ADB (adb devices shows device).
    """

    _TAG = "wE9ryARX"

    def __init__(self, port: int):
        self._proc            = None
        self._thread          = None
        self._running         = False
        self._lock            = threading.Lock()
        self._transforms      = {}
        self._buttons         = {}
        self._last_transforms = {}

    def connect(self):
        import subprocess as _sp
        # Only launch if the app isn't already streaming real pose data.
        # Re-launching via ADB while the app has XR focus kills the immersive
        # session — if the user launched it manually from inside the headset,
        # leave it alone and just attach to the logcat stream.
        probe = _sp.run(
            ["adb", "logcat", "-d", "-s", f"{self._TAG}:I"],
            capture_output=True, text=True,
        )
        already_live = any(
            self._TAG + ": " in ln
            and ln.split(self._TAG + ": ", 1)[-1].strip() not in ("", "&")
            for ln in probe.stdout.splitlines()[-20:]
        )
        if not already_live:
            _sp.run(
                ["adb", "shell", "am", "start",
                 "-n", "com.rail.oculus.teleop/.MainActivity",
                 "-c", "org.khronos.openxr.intent.category.IMMERSIVE_HMD"],
                capture_output=True,
            )
            time.sleep(1.5)
            _sp.run(["adb", "logcat", "-c"], capture_output=True)
        else:
            print("[QUEST3] OculusReader already streaming — attaching to live session")
        self._proc = _sp.Popen(
            ["adb", "logcat", "-s", f"{self._TAG}:I"],
            stdout=_sp.PIPE, stderr=_sp.DEVNULL, text=True, bufsize=1,
        )
        self._running = True
        self._thread  = threading.Thread(target=self._read_loop, daemon=True,
                                         name="quest3-logcat")
        self._thread.start()
        print("[QUEST3] Oculus Reader connected via ADB logcat — put on the headset")

    def _read_loop(self):
        for line in self._proc.stdout:
            if not self._running:
                break
            if self._TAG + ": " not in line:
                continue
            data = line.split(self._TAG + ": ", 1)[-1].strip()
            if not data or data == "&":
                continue   # headset not worn / controllers off
            transforms, buttons = self._parse_data(data)
            if transforms:
                with self._lock:
                    self._transforms = transforms
                    self._buttons    = buttons

    @staticmethod
    def _parse_data(data: str) -> tuple:
        try:
            transforms_str, buttons_str = data.split("&", 1)
        except ValueError:
            return {}, {}
        transforms = {}
        for pair_str in transforms_str.split("|"):
            if ":" not in pair_str:
                continue
            hand_char, mat_str = pair_str.split(":", 1)
            vals = mat_str.split()
            if len(vals) == 16:
                # OVR::Matrix4f::ToString outputs row-major; reshape accordingly
                mat = np.array(vals, dtype=np.float64).reshape(4, 4)
                transforms[hand_char.strip()] = mat
        try:
            from oculus_reader.buttons_parser import parse_buttons
            buttons = parse_buttons(buttons_str)
        except ImportError:
            buttons = {}
        return transforms, buttons

    def close(self):
        self._running = False
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def read_packet(self) -> dict | None:
        time.sleep(0.008)   # cap at ~125 Hz
        with self._lock:
            transforms = self._transforms
            buttons    = self._buttons
        if not transforms:
            return None
        if transforms is self._last_transforms:
            return None   # no new logcat line since last poll
        self._last_transforms = transforms
        return {"transforms": transforms, "buttons": buttons}

    @staticmethod
    def parse(raw: dict, hand: str) -> ControllerState:
        transforms = raw.get("transforms", {})
        hand_key   = hand[0]   # 'r' or 'l'
        if hand_key not in transforms:
            return ControllerState()

        mat  = transforms[hand_key]
        pos  = mat[:3, 3].copy()
        quat = _rotmat_to_quat(mat[:3, :3])

        buttons  = raw.get("buttons", {})
        trig_key = "rightTrig" if hand == "right" else "leftTrig"
        grip_key = "rightGrip" if hand == "right" else "leftGrip"
        trig_val = buttons.get(trig_key, (0.0,))
        grip_val = buttons.get(grip_key, (0.0,))
        trigger  = float(trig_val[0] if isinstance(trig_val, tuple) else trig_val)
        grip     = float(grip_val[0] if isinstance(grip_val, tuple) else grip_val)

        return ControllerState(pos=pos, quat=quat, trigger=trigger, grip=grip,
                               buttons={k: v for k, v in buttons.items()},
                               valid=True)


# ── Vuer transport (WebXR via Quest browser) ──────────────────────────────────

class _VuerTransport:
    """
    Starts a Vuer HTTPS server in a subprocess. Quest 3 browser navigates to
    https://<pc-ip>:<QUEST3_VUER_PORT>, enters an XR session, and streams
    CONTROLLER_MOVE events. Pose is written into a multiprocessing.Array so
    the main process can read it without blocking.

    SSL is mandatory for WebXR. Generate a self-signed cert once:
        openssl req -x509 -newkey rsa:4096 -nodes \\
          -out ssl/cert.pem -keyout ssl/key.pem -days 365 \\
          -subj "/CN=quest-teleop"
    Then accept the security warning in the Quest browser (once per cert).

    See docs/quest3_setup.md → Vuer section for full instructions.
    """

    def __init__(self, port: int, cert: str, key: str, hand: str):
        self._port    = port
        self._cert    = cert
        self._key     = key
        self._hand    = hand
        self._shared  = multiprocessing.Array("d", _VUER_SHARED_LEN)
        self._process = None

    def connect(self):
        self._process = multiprocessing.Process(
            target=_run_vuer_server,
            args=(self._port, self._cert, self._key, self._hand, self._shared),
            daemon=True,
        )
        self._process.start()
        time.sleep(1.5)   # let the asyncio server bind
        print(
            f"[QUEST3] WebXR server ready — open in Meta Browser:\n"
            f"  >>>    http://localhost:{self._port}\n"
            f"         Tap the VR button to enter immersive mode."
        )

    def close(self):
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)
        self._process = None

    def read_packet(self) -> dict | None:
        time.sleep(0.01)   # ~100 Hz max poll
        with self._shared.get_lock():
            valid = self._shared[9] > 0.5
            if not valid:
                return None
            data = list(self._shared)
        return {
            "pos":     data[0:3],
            "rot":     data[3:7],
            "trigger": data[7],
            "squeeze": data[8],
            "joy_x":   data[10],
            "joy_y":   data[11],
            "btn_a":   data[12],
            "btn_b":   data[13],
            "btn_x":   data[14],
            "btn_y":   data[15],
        }

    @staticmethod
    def parse(raw: dict, _hand: str) -> ControllerState:
        pos     = np.array(raw["pos"], dtype=np.float64)
        quat    = np.array(raw["rot"], dtype=np.float64)
        trigger = float(raw["trigger"])
        grip    = float(raw["squeeze"])
        joy_x   = float(raw.get("joy_x", 0.0))
        joy_y   = float(raw.get("joy_y", 0.0))
        buttons = {"A_X": float(raw.get("btn_a", 0.0)) > 0.5,
                   "B_Y": float(raw.get("btn_b", 0.0)) > 0.5,
                   "X":   float(raw.get("btn_x", 0.0)) > 0.5,
                   "Y":   float(raw.get("btn_y", 0.0)) > 0.5}
        return ControllerState(pos=pos, quat=quat, trigger=trigger, grip=grip,
                               buttons=buttons, valid=True,
                               joystick_x=joy_x, joystick_y=joy_y)


# ── UDP transport (custom streamer app) ───────────────────────────────────────

# Packet format (little-endian, 52 bytes):
#   magic      uint32   0x51455354  ("QEST")
#   timestamp  float64  seconds since epoch
#   pos_x/y/z  float32  position in metres
#   qx/qy/qz/qw float32  quaternion
#   trigger    float32  [0, 1]
#   grip       float32  [0, 1]
#   buttons    uint32   bitmask (bit 0=A/X, 1=B/Y, 2=menu, 3=thumbstick)
_UDP_MAGIC  = 0x51455354
_UDP_FMT    = "<IdfffffffI"
_UDP_SIZE   = struct.calcsize(_UDP_FMT)
_BTN_NAMES  = ["A_X", "B_Y", "menu", "thumbstick"]


class _UDPTransport:
    def __init__(self, port: int):
        self._port = port
        self._sock = None

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", self._port))
        self._sock.settimeout(0.5)
        print(f"[QUEST3] UDP transport listening on 0.0.0.0:{self._port}")

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def read_packet(self) -> dict | None:
        try:
            data, _ = self._sock.recvfrom(256)
        except socket.timeout:
            return None
        if len(data) < _UDP_SIZE:
            return None
        fields = struct.unpack_from(_UDP_FMT, data)
        magic, ts, px, py, pz, qx, qy, qz, qw, trig, grip, btn_mask = fields
        if magic != _UDP_MAGIC:
            return None
        return dict(
            pos=[px, py, pz], rot=[qx, qy, qz, qw],
            trigger=trig, grip=grip,
            buttons={name: bool(btn_mask & (1 << i)) for i, name in enumerate(_BTN_NAMES)},
        )

    @staticmethod
    def parse(raw: dict, _hand: str) -> ControllerState:
        pos     = np.array(raw["pos"], dtype=np.float64)
        quat    = np.array(raw["rot"], dtype=np.float64)
        trigger = float(raw.get("trigger", 0.0))
        grip    = float(raw.get("grip",    0.0))
        buttons = raw.get("buttons", {})
        return ControllerState(pos=pos, quat=quat, trigger=trigger, grip=grip,
                               buttons=buttons, valid=True)


# ── Public reader class ───────────────────────────────────────────────────────

class Quest3Reader:
    """
    Thread-safe Quest 3 controller reader.

    Usage (context manager):
        with Quest3Reader() as quest:
            state = quest.get_state()   # ControllerState
    """

    def __init__(self):
        mode = QUEST3_MODE
        if mode == "oculus_reader":
            self._transport = _OculusReaderTransport(QUEST3_ADB_PORT)
        elif mode == "vuer":
            self._transport = _VuerTransport(
                QUEST3_VUER_PORT, QUEST3_VUER_CERT, QUEST3_VUER_KEY, QUEST3_ACTIVE_HAND,
            )
        elif mode == "udp":
            self._transport = _UDPTransport(QUEST3_UDP_PORT)
        else:
            raise ValueError(
                f"Unknown QUEST3_MODE {mode!r} — expected 'oculus_reader', 'vuer', or 'udp'"
            )

        self._hand     = QUEST3_ACTIVE_HAND
        self._state    = ControllerState()
        self._last_ts  = 0.0
        self._lock     = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread   = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def open(self):
        self._transport.connect()
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="quest3-reader"
        )
        self._thread.start()

        if QUEST3_MODE == "oculus_reader":
            # Wait indefinitely — print a reminder every 10 s so the user knows
            # to put on the headset. No timeout: the user picks up the headset
            # whenever they are ready.
            print("[QUEST3] Waiting for controller data — put on the headset now...")
            last_reminder = time.monotonic()
            while True:
                with self._lock:
                    if self._state.valid:
                        break
                now = time.monotonic()
                if now - last_reminder >= 10.0:
                    print("[QUEST3] Still waiting — put on the headset and pick up the right controller...")
                    last_reminder = now
                time.sleep(0.05)
        else:
            if QUEST3_MODE == "vuer":
                print("[QUEST3] Waiting for Quest 3 to connect to Vuer server...")
                print("         Navigate the Quest 3 browser to the URL shown above.")
                deadline = time.monotonic() + 600.0
            else:
                deadline = time.monotonic() + 5.0

            while time.monotonic() < deadline:
                with self._lock:
                    if self._state.valid:
                        break
                time.sleep(0.05)
            else:
                raise TimeoutError(
                    "[QUEST3] No data received within timeout.\n"
                    + self._timeout_hint()
                )

    def _timeout_hint(self) -> str:
        if QUEST3_MODE == "oculus_reader":
            return ("         Check: Quest 3 is connected via USB (adb devices)\n"
                    "                OculusReader app is running on the headset (put it on)")
        if QUEST3_MODE == "vuer":
            return ("         Check: Quest 3 browser opened the Vuer URL\n"
                    "                SSL cert warning was accepted\n"
                    "                'Enter VR' was tapped")
        return "         Check: Quest 3 streamer app is running and pointing to this PC"

    def close(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._transport.close()

    # ── public API ────────────────────────────────────────────────────────────

    def get_state(self) -> ControllerState:
        """Return a copy of the latest ControllerState. valid=False if data is stale."""
        with self._lock:
            state = ControllerState(
                pos=self._state.pos.copy(),
                quat=self._state.quat.copy(),
                trigger=self._state.trigger,
                grip=self._state.grip,
                buttons=dict(self._state.buttons),
                valid=self._state.valid,
                joystick_x=self._state.joystick_x,
                joystick_y=self._state.joystick_y,
            )
        if (time.time() - self._last_ts) * 1000 > QUEST3_STALE_MS:
            state.valid = False
        return state

    # ── background thread ─────────────────────────────────────────────────────

    def _loop(self):
        consecutive_errors = 0
        while not self._stop_evt.is_set():
            try:
                raw = self._transport.read_packet()
                if raw is None:
                    continue
                state = self._transport.parse(raw, self._hand)
                with self._lock:
                    self._state   = state
                    self._last_ts = time.time()
                consecutive_errors = 0
            except (ConnectionResetError, BrokenPipeError) as exc:
                if self._stop_evt.is_set():
                    break
                print(f"[QUEST3] Connection lost ({exc}) — reconnecting in 1 s")
                self._transport.close()
                time.sleep(1.0)
                try:
                    self._transport.connect()
                except Exception as e:
                    print(f"[QUEST3] Reconnect failed: {e}")
            except Exception as exc:
                if self._stop_evt.is_set():
                    break
                consecutive_errors += 1
                if consecutive_errors == 1 or consecutive_errors % 30 == 0:
                    print(f"[QUEST3] Read error #{consecutive_errors}: {exc!r}")

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
