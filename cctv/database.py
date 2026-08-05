"""
cctv/database.py — SQLite persistence for the CCTV model.

Stores per-session analytics so cross-video temporal trends can be
computed (count over days, activity patterns, recurring anomalies).

Schema
------
sessions     — one row per processed video
tracks       — per-cow summary for each session
density_maps — serialised grid per session
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from cctv.config import DB_PATH
from cctv.analytics import AnalyticsResult
from cctv.pipeline import VideoSummary


def _get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _tx(db_path: Path = DB_PATH):
    conn = _get_conn(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── schema ────────────────────────────────────────────────────────

def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if they don't exist."""
    with _tx(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                job_id          TEXT PRIMARY KEY,
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                location_tag    TEXT,
                video_filename  TEXT,
                final_cattle_count  INTEGER,
                count_method    TEXT,
                max_in_frame    INTEGER,
                avg_confidence  REAL,
                total_detections INTEGER,
                throughput_fps  REAL,
                processing_sec  REAL,
                source_fps      REAL,
                source_width    INTEGER,
                source_height   INTEGER,
                total_frames    INTEGER,
                frames_processed INTEGER,
                frames_with_cattle INTEGER,
                avg_herd_speed  REAL,
                isolated_cattle TEXT,
                activity_breakdown TEXT,
                frame_count_series TEXT,
                summary_text    TEXT,
                output_video    TEXT,
                output_report   TEXT,
                output_csv      TEXT
            );

            CREATE TABLE IF NOT EXISTS tracks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id          TEXT NOT NULL REFERENCES sessions(job_id),
                stable_id       INTEGER NOT NULL,
                frames_visible  INTEGER,
                total_distance_px REAL,
                avg_speed       REAL,
                max_speed       REAL,
                avg_bbox_area   REAL,
                avg_nn_distance REAL,
                is_isolated     INTEGER,
                activity        TEXT,
                trajectory_json TEXT,
                dwell_zones_json TEXT
            );

            CREATE TABLE IF NOT EXISTS density_maps (
                job_id          TEXT PRIMARY KEY REFERENCES sessions(job_id),
                grid_json       TEXT NOT NULL,
                grid_cells      INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_tracks_job ON tracks(job_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
            CREATE INDEX IF NOT EXISTS idx_sessions_location ON sessions(location_tag);
        """)


# ── write ─────────────────────────────────────────────────────────

def save_session(
    summary: VideoSummary,
    analytics: AnalyticsResult,
    location_tag: Optional[str] = None,
    video_filename: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> None:
    """Persist a completed pipeline run + analytics to the database."""
    init_db(db_path)

    with _tx(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO sessions (
                job_id, location_tag, video_filename,
                final_cattle_count, count_method, max_in_frame,
                avg_confidence, total_detections, throughput_fps,
                processing_sec, source_fps, source_width, source_height,
                total_frames, frames_processed, frames_with_cattle,
                avg_herd_speed, isolated_cattle, activity_breakdown,
                frame_count_series, summary_text,
                output_video, output_report, output_csv
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            summary.job_id, location_tag, video_filename,
            # Peak-in-frame, matching what /result and /analytics now report
            # (see CLAUDE.md) -- was summary.final_cattle_count (tracked-ID
            # count), which is why /history and /trends used to disagree
            # with /result for the same job.
            summary.max_cattle_in_frame, summary.count_method,
            summary.max_cattle_in_frame, summary.average_confidence,
            summary.total_detections, summary.throughput_fps,
            summary.processing_seconds, summary.source_fps,
            summary.source_width, summary.source_height,
            summary.total_frames, summary.frames_processed,
            summary.frames_with_cattle,
            analytics.avg_herd_speed,
            json.dumps(analytics.isolated_cattle),
            json.dumps(analytics.activity_breakdown),
            json.dumps(analytics.frame_count_series),
            analytics.summary_text,
            summary.output_video, summary.output_report, summary.output_csv,
        ))

        # tracks
        for cow in analytics.per_cow:
            conn.execute("""
                INSERT INTO tracks (
                    job_id, stable_id, frames_visible,
                    total_distance_px, avg_speed, max_speed,
                    avg_bbox_area, avg_nn_distance, is_isolated,
                    activity, trajectory_json, dwell_zones_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                summary.job_id, cow.stable_id, cow.frames_visible,
                cow.total_distance_px, cow.avg_speed_px_per_frame,
                cow.max_speed_px_per_frame, cow.avg_bbox_area,
                cow.avg_nearest_neighbour_px, int(cow.is_isolated),
                cow.activity,
                json.dumps(cow.trajectory),
                json.dumps(cow.dwell_zones),
            ))

        # density map
        conn.execute("""
            INSERT OR REPLACE INTO density_maps (job_id, grid_json, grid_cells)
            VALUES (?,?,?)
        """, (
            summary.job_id,
            json.dumps(analytics.density_grid),
            len(analytics.density_grid),
        ))


# ── read ──────────────────────────────────────────────────────────

def list_sessions(
    location_tag: Optional[str] = None,
    limit: int = 50,
    db_path: Path = DB_PATH,
) -> list[dict]:
    """Return recent sessions, newest first."""
    init_db(db_path)
    with _tx(db_path) as conn:
        if location_tag:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE location_tag=? "
                "ORDER BY created_at DESC LIMIT ?",
                (location_tag, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_session(job_id: str, db_path: Path = DB_PATH) -> Optional[dict]:
    init_db(db_path)
    with _tx(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE job_id=?", (job_id,)
        ).fetchone()
    return dict(row) if row else None


def get_tracks(job_id: str, db_path: Path = DB_PATH) -> list[dict]:
    init_db(db_path)
    with _tx(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM tracks WHERE job_id=? ORDER BY stable_id",
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_density_map(job_id: str, db_path: Path = DB_PATH) -> Optional[list]:
    init_db(db_path)
    with _tx(db_path) as conn:
        row = conn.execute(
            "SELECT grid_json FROM density_maps WHERE job_id=?", (job_id,)
        ).fetchone()
    return json.loads(row["grid_json"]) if row else None


def get_trend_data(
    location_tag: Optional[str] = None,
    limit: int = 100,
    db_path: Path = DB_PATH,
) -> list[dict]:
    """Return (date, count, avg_speed, isolated_count) for trend charts."""
    init_db(db_path)
    with _tx(db_path) as conn:
        query = """
            SELECT
                created_at, final_cattle_count, avg_herd_speed,
                isolated_cattle, location_tag, job_id
            FROM sessions
        """
        params: list = []
        if location_tag:
            query += " WHERE location_tag=?"
            params.append(location_tag)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()

    results = []
    for r in rows:
        iso = json.loads(r["isolated_cattle"]) if r["isolated_cattle"] else []
        results.append({
            "date": r["created_at"],
            "count": r["final_cattle_count"],
            "avg_speed": r["avg_herd_speed"],
            "isolated_count": len(iso),
            "location": r["location_tag"],
            "job_id": r["job_id"],
        })
    return results
