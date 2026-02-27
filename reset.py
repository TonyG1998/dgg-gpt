import os
import sqlite3
from dotenv import load_dotenv
from logger import get_logger

load_dotenv()
log = get_logger("reset")

DB_PATH = os.getenv("SQLITE_DB", "/data/streams/streams.db")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")


def reset(preserve_enrolled: bool = True):
    # ── SQLite ────────────────────────────────────────────────────────────────
    log.info("Clearing SQLite RAG tables...")
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")

        # ingest_log and speaker_global_map always get cleared
        for table in ["ingest_log", "speaker_global_map"]:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            conn.execute(f"DELETE FROM {table}")
            log.info(f"  Cleared {table}: {count} rows deleted")

        # global_speakers — optionally preserve enrolled speakers
        if preserve_enrolled:
            # Enrolled speakers have global_ids stored in Qdrant with is_enrolled=True
            # Keep them in SQLite by only deleting non-enrolled ones
            # We identify enrolled ones from Qdrant below, but for SQLite
            # we keep ALL global_speakers since ingest will re-use them by global_id
            total = conn.execute(
                "SELECT COUNT(*) FROM global_speakers").fetchone()[0]
            log.info(f"  Preserving global_speakers ({
                     total} entries kept for re-use)")
        else:
            count = conn.execute(
                "SELECT COUNT(*) FROM global_speakers").fetchone()[0]
            conn.execute("DELETE FROM global_speakers")
            log.info(f"  Cleared global_speakers: {count} rows deleted")

        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()

        print("\nSQLite after reset:")
        for table in ["ingest_log", "global_speakers", "speaker_global_map"]:
            remaining = conn.execute(
                f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            status = "✓" if table == "global_speakers" and preserve_enrolled \
                     else ("✓" if remaining == 0 else "✗ STILL HAS DATA")
            print(f"  {table:<25} {remaining} rows {status}")

    except Exception as e:
        conn.rollback()
        log.error(f"SQLite reset failed: {e}")
        raise
    finally:
        conn.close()

    # ── Qdrant ────────────────────────────────────────────────────────────────
    log.info("Clearing Qdrant collections...")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        q = QdrantClient(host=QDRANT_HOST, port=6333)
        existing = [c.name for c in q.get_collections().collections]

        print("\nQdrant after reset:")

        # transcripts — always wipe completely
        if "transcripts" in existing:
            q.delete_collection("transcripts")
            log.info("  Deleted Qdrant 'transcripts' collection")
            print(f"  {'transcripts':<20} ✓ deleted")
        else:
            print(f"  {'transcripts':<20} (did not exist)")

        # speakers — delete only non-enrolled points if preserve_enrolled
        if "speakers" in existing:
            if preserve_enrolled:
                # Count enrolled vs total
                all_points, _ = q.scroll(
                    "speakers", limit=1000, with_payload=True)
                enrolled = [
                    p for p in all_points if p.payload.get("is_enrolled")]
                non_enrolled = [
                    p for p in all_points if not p.payload.get("is_enrolled")]

                if non_enrolled:
                    q.delete(
                        collection_name="speakers",
                        points_selector=[p.id for p in non_enrolled]
                    )
                    log.info(f"  Deleted {len(non_enrolled)
                                          } non-enrolled speaker points")

                print(f"  {'speakers':<20} ✓ kept {len(enrolled)} enrolled, "
                      f"deleted {len(non_enrolled)} auto-generated")
                for p in enrolled:
                    print(f"    → preserved: {p.payload['display_name']} "
                          f"({p.payload['global_id']})")
            else:
                q.delete_collection("speakers")
                log.info("  Deleted Qdrant 'speakers' collection entirely")
                print(f"  {'speakers':<20} ✓ deleted entirely")

    except Exception as e:
        log.error(f"Qdrant reset failed: {e}")
        raise

    print("\n✓ Reset complete — safe to run python ingest.py\n")
    if preserve_enrolled:
        print("  Enrolled speakers were preserved and will be matched during ingest.")
    log.info("Reset complete")


if __name__ == "__main__":
    import sys

    force = "--confirm" in sys.argv
    # add this flag to wipe everything including enrollments
    full_wipe = "--full" in sys.argv

    if not force:
        print("⚠ This will delete all ingested data from SQLite and Qdrant.")
        if not full_wipe:
            print("  Enrolled speakers (is_enrolled=True) will be PRESERVED.")
        else:
            print("  --full flag detected: enrolled speakers will also be deleted.")
        print("  vods and transcript_segments will NOT be touched.")
        print("")
        confirm = input("Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    reset(preserve_enrolled=not full_wipe)
