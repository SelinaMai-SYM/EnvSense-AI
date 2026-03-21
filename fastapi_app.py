from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dashboard.router import run_pipeline
from utils.io_utils import repo_root
from utils.time_utils import now_iso


Mode = Literal["study", "sleep"]
Source = Literal["realtime", "offline"]
MorningFeedback = Literal["slept_well", "okay", "poor_sleep"]


class LatestRow(BaseModel):
    temp_C: float
    humidity: float
    eco2_ppm: float
    tvoc: float


class ViewResponse(BaseModel):
    latest: LatestRow | None
    last_updated: str | None
    prediction: Dict[str, Any]
    trends: Dict[str, Any]


class MorningFeedbackIn(BaseModel):
    morning_feedback: MorningFeedback
    # Optional grouping for multiple feedbacks in the same "night" session.
    # If omitted, the server will create a new session folder automatically.
    session_name: Optional[str] = None


def _sanitize_session_name(name: str) -> str:
    name = name.strip()
    # Disallow path traversal / nested folders.
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("invalid session_name")
    # Keep it simple for filesystem compatibility.
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", name)
    return name[:120] if len(name) > 120 else name


def _resolve_csv(*, mode: Mode, source: Source) -> Path:
    root = repo_root()
    if source == "realtime":
        return root / "data" / "realtime.csv"

    # Offline demo CSVs (replay recordings).
    if mode == "sleep":
        return root / "data" / "sleep_sessions" / "bedroom" / "realtime_bedroom.csv"
    return root / "data" / "room_reset_sessions" / "classroom" / "realtime_classroom.csv"


def _latest_from_df(df: pd.DataFrame) -> tuple[LatestRow | None, str | None]:
    if df is None or df.empty or "timestamp" not in df.columns:
        return None, None
    df = df.sort_values("timestamp")
    last = df.iloc[-1]
    try:
        return (
            LatestRow(
                temp_C=float(last["temp_C"]),
                humidity=float(last["humidity"]),
                eco2_ppm=float(last["eco2_ppm"]),
                tvoc=float(last["tvoc"]),
            ),
            str(pd.to_datetime(last["timestamp"], utc=True).isoformat()),
        )
    except Exception:
        return None, None


def _df_to_trends(df_window: pd.DataFrame) -> Dict[str, Any]:
    if df_window is None or df_window.empty or "timestamp" not in df_window.columns:
        return {"timestamps": [], "temp_C": [], "humidity": [], "eco2_ppm": [], "tvoc": []}

    df_window = df_window.sort_values("timestamp")
    ts = pd.to_datetime(df_window["timestamp"], utc=True, errors="coerce")
    timestamps = [t.isoformat() if t is not pd.NaT else None for t in ts.tolist()]

    def series(col: str) -> list[float | None]:
        if col not in df_window.columns:
            return [None] * len(df_window)
        vals = pd.to_numeric(df_window[col], errors="coerce").tolist()
        out: list[float | None] = []
        for v in vals:
            if v is None:
                out.append(None)
            else:
                try:
                    f = float(v)
                    out.append(f if f == f else None)  # NaN-safe
                except Exception:
                    out.append(None)
        return out

    return {
        "timestamps": timestamps,
        "temp_C": series("temp_C"),
        "humidity": series("humidity"),
        "eco2_ppm": series("eco2_ppm"),
        "tvoc": series("tvoc"),
    }


def create_app() -> FastAPI:
    app = FastAPI(title="EnvSense AI (FastAPI)")

    root = repo_root()
    frontend_dir = root / "frontend"
    mount_frontend = frontend_dir.exists()

    @app.get("/api/view", response_model=ViewResponse)
    def api_view(
        mode: Mode = Query("study", description="study | sleep"),
        source: Source = Query("realtime", description="realtime (sensor) | offline (demo)"),
    ) -> Any:
        csv_path = _resolve_csv(mode=mode, source=source)
        try:
            prediction, df_window = run_pipeline(mode=mode, csv_path=csv_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e

        latest, last_updated = _latest_from_df(df_window)
        trends = _df_to_trends(df_window)

        return ViewResponse(
            latest=latest,
            last_updated=last_updated,
            prediction=prediction,
            trends=trends,
        )

    @app.post("/api/sleep/morning-feedback")
    def api_morning_feedback(payload: MorningFeedbackIn = Body(...)) -> JSONResponse:
        root = repo_root()
        sleep_sessions_root = root / "data" / "sleep_sessions"
        sleep_sessions_root.mkdir(parents=True, exist_ok=True)

        if payload.session_name:
            try:
                session_name = _sanitize_session_name(payload.session_name)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid session_name")
        else:
            # Auto session folder (one feedback event).
            session_name = now_iso().replace(":", "-")
            session_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", session_name)

        session_dir = sleep_sessions_root / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        fp = session_dir / "morning_feedback.csv"
        write_header = not fp.exists() or fp.stat().st_size == 0
        try:
            with fp.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["timestamp", "morning_feedback"])
                writer.writerow([now_iso(), payload.morning_feedback])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"write failed: {e}") from e

        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "session_name": session_name,
                "timestamp": now_iso(),
                "morning_feedback": payload.morning_feedback,
            },
        )

    @app.get("/")
    def index_fallback() -> FileResponse:
        # If the static mount is missing for some reason, still respond cleanly.
        idx = frontend_dir / "index.html"
        if not idx.exists():
            return FileResponse(idx)
        return FileResponse(str(idx))

    # Important: mount static files AFTER API routes, so `/api/*` doesn't get intercepted.
    if mount_frontend:
        # Serve the SPA at `/` (e.g. `/` -> index.html, `/app.js`, `/styles.css`, ...).
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


app = create_app()

