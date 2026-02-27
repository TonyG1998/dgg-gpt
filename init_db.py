import sqlite3
import os
import glob
import re
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB", "/data/streams/streams.db")
SQL_DIR = os.getenv("SQL_DIR", "/data/streams/sql")

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS vods (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  date TEXT NOT NULL,
  video_id TEXT NOT NULL,
  has_transcript INTEGER NOT NULL DEFAULT 0,
  duration_seconds INTEGER
);

CREATE TABLE IF NOT EXISTS speakers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT
);

CREATE TABLE IF NOT EXISTS transcript_segments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  vod_id INTEGER NOT NULL REFERENCES vods(id),
  start_time REAL NOT NULL,
  text TEXT NOT NULL,
  speaker_id INTEGER REFERENCES speakers(id),
  speaker_confidence REAL,
  diarization_label TEXT
);

CREATE TABLE IF NOT EXISTS global_speakers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  global_id TEXT NOT NULL UNIQUE,
  display_name TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS speaker_global_map (
  speaker_id INTEGER NOT NULL,
  vod_id INTEGER NOT NULL,
  global_speaker_id INTEGER NOT NULL REFERENCES global_speakers(id),
  confidence REAL,
  PRIMARY KEY (speaker_id, vod_id),
  FOREIGN KEY (speaker_id) REFERENCES speakers(id),
  FOREIGN KEY (vod_id) REFERENCES vods(id)
);

CREATE TABLE IF NOT EXISTS ingest_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  vod_id INTEGER NOT NULL UNIQUE REFERENCES vods(id),
  ingested_at TEXT DEFAULT (datetime('now')),
  chunk_count INTEGER
);

CREATE INDEX IF NOT EXISTS idx_segments_vod     ON transcript_segments(vod_id);
CREATE INDEX IF NOT EXISTS idx_segments_speaker ON transcript_segments(speaker_id);
CREATE INDEX IF NOT EXISTS idx_map_vod          ON speaker_global_map(vod_id);
CREATE INDEX IF NOT EXISTS idx_map_speaker      ON speaker_global_map(speaker_id);
"""


def init_db():
    print(f"Initializing database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    print("✓ Schema created")
    return conn


def import_sql_files(conn: sqlite3.Connection):
    """
    Imports INSERT data from .sql files, skipping CREATE TABLE / DROP / PRAGMA
    statements that would conflict with the schema already created by init_db().
    Files are processed alphabetically — prefix with 01_, 02_ etc. if order matters.
    """
    sql_files = sorted(glob.glob(os.path.join(SQL_DIR, "*.sql")))
    if not sql_files:
        print(f"No .sql files found in {SQL_DIR} — skipping import.")
        return

    # Regex to match statement types we want to skip
    skip_pattern = re.compile(
        r'^\s*(CREATE|DROP|ALTER|PRAGMA|BEGIN|COMMIT|ROLLBACK)',
        re.IGNORECASE
    )

    for path in sql_files:
        filename = os.path.basename(path)
        print(f"Importing {filename}...")

        with open(path, "r") as f:
            raw = f.read()

        # Split into individual statements on semicolons
        statements = [s.strip() for s in raw.split(";") if s.strip()]
        insert_statements = [
            s for s in statements if not skip_pattern.match(s)
        ]

        if not insert_statements:
            print(f"  ⚠ No INSERT statements found in {filename} — skipping.")
            continue

        success, skipped, errors = 0, 0, 0
        for stmt in insert_statements:
            try:
                conn.execute(stmt)
                success += 1
            except sqlite3.Error as e:
                print(f"  ✗ Statement error: {e}")
                print(f"    Statement: {stmt[:80]}...")
                errors += 1

        conn.commit()
        print(f"  ✓ {filename}: {success} inserted, {errors} errors")


def verify_db(conn: sqlite3.Connection):
    tables = ["vods", "speakers", "transcript_segments",
              "global_speakers", "speaker_global_map", "ingest_log"]
    print("\nRow counts after import:")
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<30} {count:>6} rows")


if __name__ == "__main__":
    conn = init_db()
    import_sql_files(conn)
    verify_db(conn)
    conn.close()
    print(f"\n✓ Database ready at {DB_PATH}")
