"""Engine <-> UI communication via small files in storage/.

The engine periodically writes a JSON snapshot of its state; the Streamlit
UI reads it. The UI stops the engine by creating a STOP flag file, which the
engine checks every cycle. File-based signalling keeps the two processes
fully decoupled and works on every OS.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from config.settings import STORAGE_DIR

STATE_PATH = STORAGE_DIR / "engine_state.json"
PID_PATH = STORAGE_DIR / "engine.pid"
STOP_PATH = STORAGE_DIR / "STOP"


def write_state(state: dict) -> None:
    STORAGE_DIR.mkdir(exist_ok=True)
    state = {**state, "updated_at": datetime.now(timezone.utc).isoformat()}
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(STATE_PATH)  # atomic on POSIX and Windows


def read_state() -> dict | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return None


def write_pid() -> None:
    STORAGE_DIR.mkdir(exist_ok=True)
    PID_PATH.write_text(str(os.getpid()))


def clear_pid() -> None:
    PID_PATH.unlink(missing_ok=True)


def engine_running() -> bool:
    """True if a PID file exists and that process is still alive."""
    if not PID_PATH.exists():
        return False
    try:
        pid = int(PID_PATH.read_text().strip())
        os.kill(pid, 0)  # signal 0: existence check only
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        return False


def request_stop() -> None:
    STORAGE_DIR.mkdir(exist_ok=True)
    STOP_PATH.touch()


def stop_requested() -> bool:
    return STOP_PATH.exists()


def clear_stop() -> None:
    STOP_PATH.unlink(missing_ok=True)


__all__ = ["write_state", "read_state", "write_pid", "clear_pid",
           "engine_running", "request_stop", "stop_requested", "clear_stop",
           "STATE_PATH", "PID_PATH", "STOP_PATH"]
