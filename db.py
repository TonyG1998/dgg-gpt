import sqlite3
import os
from contextlib import contextmanager
from dotenv import load_dotenv
from logger import get_logger

load_dotenv()
log = get_logger("db")
DB_PATH = os.getenv("SQLITE_DB", "/data/streams/streams.db")


@contextmanager
def get_conn():
    log.debug(f"Opening DB connection to {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
        log.debug("DB transaction committed")
    except Exception as e:
        conn.rollback()
        log.error(f"DB transaction rolled back due to error: {e}")
        raise
    finally:
        conn.close()
        log.debug("DB connection closed")

# ── VODs ──────────────────────────────────────────────────────────────────────


def get_all_vods() -> list:
    log.debug("Fetching all vods")
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM vods ORDER BY date DESC").fetchall()
    log.debug(f"Found {len(rows)} vods")
    return rows


def get_vod_by_id(vod_id: int):
    log.debug(f"Fetching vod id={vod_id}")
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM vods WHERE id = ?",
                           (vod_id,)).fetchone()
    if row:
        log.debug(f"Found vod: '{row['title']}'")
    else:
        log.warning(f"No vod found with id={vod_id}")
    return row


def get_vods_pending_ingest() -> list:
    log.debug("Fetching vods pending ingest")
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT v.* FROM vods v
            WHERE v.has_transcript = 1
            AND v.id NOT IN (SELECT vod_id FROM ingest_log)
            ORDER BY v.date ASC
        """).fetchall()
    log.info(f"Found {len(rows)} vod(s) pending ingest")
    return rows

# ── Segments ──────────────────────────────────────────────────────────────────


def get_segments_for_vod(vod_id: int) -> list:
    log.debug(f"Fetching segments for vod_id={vod_id}")
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT id, vod_id, start_time, text,
                   speaker_id, speaker_confidence, diarization_label
            FROM transcript_segments
            WHERE vod_id = ?
            ORDER BY start_time ASC
        """, (vod_id,)).fetchall()
    log.debug(f"Found {len(rows)} segments for vod_id={vod_id}")
    return rows


def get_unique_speakers_for_vod(vod_id: int) -> list:
    log.debug(f"Fetching unique speakers for vod_id={vod_id}")
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT diarization_label,
                   MIN(start_time) AS first_seen
            FROM transcript_segments
            WHERE vod_id = ? AND diarization_label IS NOT NULL
            AND diarization_label != 'SPEAKER_UNKNOWN'
            GROUP BY diarization_label
            ORDER BY first_seen ASC
        """, (vod_id,)).fetchall()
    log.debug(f"Found {len(rows)} unique speaker labels for vod_id={vod_id}")
    return rows

# ── Global speakers ───────────────────────────────────────────────────────────


def get_global_speaker_map_for_vod(vod_id: int) -> dict:
    log.debug(f"Fetching global speaker map for vod_id={vod_id}")
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT sm.diarization_label, gs.global_id, gs.display_name
            FROM speaker_global_map sm
            JOIN global_speakers gs ON sm.global_speaker_id = gs.id
            WHERE sm.vod_id = ?
        """, (vod_id,)).fetchall()
    result = {r["diarization_label"]: (
        r["global_id"], r["display_name"]) for r in rows}
    log.debug(f"Global speaker map for vod_id={vod_id}: {result}")
    return result


def upsert_global_speaker(global_id: str, display_name: str) -> int:
    log.debug(f"Upserting global speaker global_id={
              global_id}, display_name='{display_name}'")
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO global_speakers (global_id, display_name)
            VALUES (?, ?)
            ON CONFLICT(global_id) DO UPDATE SET display_name = excluded.display_name
        """, (global_id, display_name))
        db_id = conn.execute(
            "SELECT id FROM global_speakers WHERE global_id = ?", (global_id,)
        ).fetchone()["id"]
    log.debug(f"Global speaker upserted with db_id={db_id}")
    return db_id


def save_speaker_mapping(diarization_label: str, vod_id: int,
                         global_speaker_db_id: int, confidence: float):
    log.debug(f"Saving speaker mapping: '{diarization_label}' vod_id={vod_id} "
              f"→ global_speaker_db_id={global_speaker_db_id} confidence={confidence:.3f}")
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO speaker_global_map
                (diarization_label, vod_id, global_speaker_id, confidence)
            VALUES (?, ?, ?, ?)
        """, (diarization_label, vod_id, global_speaker_db_id, confidence))
    log.debug("Speaker mapping saved")


def list_global_speakers() -> list:
    log.debug("Listing all global speakers")
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM global_speakers ORDER BY display_name"
        ).fetchall()
    log.debug(f"Found {len(rows)} global speakers")
    return rows


def rename_global_speaker(global_id: str, new_name: str):
    log.info(f"Renaming global speaker {global_id} → '{new_name}'")
    with get_conn() as conn:
        conn.execute(
            "UPDATE global_speakers SET display_name = ? WHERE global_id = ?",
            (new_name, global_id)
        )
    log.info(f"Rename complete for {global_id}")

# ── Ingest log ────────────────────────────────────────────────────────────────


def mark_vod_ingested(vod_id: int, chunk_count: int):
    log.debug(f"Marking vod_id={vod_id} as ingested with {chunk_count} chunks")
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO ingest_log (vod_id, chunk_count)
            VALUES (?, ?)
        """, (vod_id, chunk_count))
    log.info(f"VOD {vod_id} marked as ingested ({chunk_count} chunks)")


def get_ingest_status() -> list:
    log.debug("Fetching ingest status for all vods")
    with get_conn() as conn:
        return conn.execute("""
            SELECT v.id, v.title, v.date, v.has_transcript,
                   il.ingested_at, il.chunk_count
            FROM vods v
            LEFT JOIN ingest_log il ON v.id = il.vod_id
            ORDER BY v.date DESC
        """).fetchall()
