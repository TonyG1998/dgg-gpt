import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
from logger import get_logger
import db

load_dotenv()
log = get_logger("verify_speakers")
qdrant = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)


# ── 1. Show how each diarization label mapped across VODs ─────────────────────

def show_global_speaker_coverage():
    """
    Show all global speakers sorted by total segment count (most prevalent first).
    Includes per-VOD breakdown with diarization labels and confidence.
    """
    with db.get_conn() as conn:
        # Get total segment count per global speaker for sorting
        totals = conn.execute("""
            SELECT
                gs.global_id,
                gs.display_name,
                COUNT(ts.id) as total_segments,
                COUNT(DISTINCT sgm.vod_id) as total_vods,
                ROUND(COUNT(ts.id) * 100.0 / (
                    SELECT COUNT(*) FROM transcript_segments
                ), 1) as pct_of_all
            FROM global_speakers gs
            JOIN speaker_global_map sgm ON sgm.global_speaker_id = gs.id
            JOIN transcript_segments ts
                ON ts.vod_id = sgm.vod_id
                AND ts.diarization_label = sgm.diarization_label
            GROUP BY gs.global_id
            ORDER BY total_segments DESC
        """).fetchall()

        if not totals:
            print("No global speaker mappings found. Run ingest first.")
            return

        # Get per-VOD breakdown
        breakdowns = conn.execute("""
            SELECT
                gs.global_id,
                v.id as vod_id,
                v.title,
                v.date,
                sgm.diarization_label,
                sgm.confidence,
                COUNT(ts.id) as segment_count
            FROM global_speakers gs
            JOIN speaker_global_map sgm ON sgm.global_speaker_id = gs.id
            JOIN vods v ON sgm.vod_id = v.id
            JOIN transcript_segments ts
                ON ts.vod_id = v.id
                AND ts.diarization_label = sgm.diarization_label
            GROUP BY gs.global_id, v.id, sgm.diarization_label
            ORDER BY gs.global_id, v.date ASC
        """).fetchall()

    # Group breakdowns by global_id
    by_speaker = {}
    for r in breakdowns:
        gid = r["global_id"]
        if gid not in by_speaker:
            by_speaker[gid] = []
        by_speaker[gid].append(r)

    print(f"\n{'Rank':<6} {'Display Name':<25} {'Total Segs':>11} "
          f"{'VODs':>6} {'% All Segs':>11}")
    print("=" * 65)

    for rank, t in enumerate(totals, 1):
        named = t["display_name"] if not t["display_name"].startswith("spk_") \
            else "⚠ NOT NAMED"
        print(f"\n{rank:<6} {named:<25} {t['total_segments']:>11,} "
              f"{t['total_vods']:>6} {t['pct_of_all']:>10.1f}%")
        print(f"       ({t['global_id']})")
        print(f"       {'VOD ID':<8} {'Date':<12} {'Label':<14} "
              f"{'Segments':>9} {'Confidence':>11}")
        print(f"       {'-'*56}")

        for r in by_speaker.get(t["global_id"], []):
            conf = f"{r['confidence']:.3f}" if r["confidence"] else "manual"
            print(f"       {r['vod_id']:<8} {r['date']:<12} "
                  f"{r['diarization_label']:<14} "
                  f"{r['segment_count']:>9,} {conf:>11}")


def show_cross_vod_samples(global_id: str, samples_per_vod: int = 3):
    """
    Pull sample transcript lines for a global speaker from each VOD,
    with clickable YouTube timestamp URLs for manual verification.
    Sorted by VOD date ascending.
    """
    with db.get_conn() as conn:
        mappings = conn.execute("""
            SELECT sgm.vod_id, sgm.diarization_label,
                   v.title, v.date, v.video_id
            FROM speaker_global_map sgm
            JOIN global_speakers gs ON sgm.global_speaker_id = gs.id
            JOIN vods v ON sgm.vod_id = v.id
            WHERE gs.global_id = ?
            ORDER BY v.date ASC
        """, (global_id,)).fetchall()

        if not mappings:
            print(f"No mappings found for global_id={global_id}")
            return

        speaker_name = conn.execute(
            "SELECT display_name FROM global_speakers WHERE global_id = ?",
            (global_id,)
        ).fetchone()["display_name"]

    print(f"\nCross-VOD verification: {speaker_name} ({global_id})")
    print(f"Appears in {len(mappings)} VOD(s)\n")

    with db.get_conn() as conn:
        for m in mappings:
            print(f"{'='*65}")
            print(f"VOD {m['vod_id']}: {m['title']}")
            print(f"Date: {m['date']} | Label: {m['diarization_label']}")
            print(f"{'-'*65}")

            samples = conn.execute("""
                SELECT start_time, text
                FROM transcript_segments
                WHERE vod_id = ? AND diarization_label = ?
                AND LENGTH(TRIM(text)) > 40
                ORDER BY RANDOM()
                LIMIT ?
            """, (m["vod_id"], m["diarization_label"],
                  samples_per_vod)).fetchall()

            if not samples:
                print("  (no qualifying segments found)")
                continue

            for s in samples:
                start_sec = int(s["start_time"])
                h = start_sec // 3600
                mins = (start_sec % 3600) // 60
                sec = start_sec % 60
                ts = f"{h:02d}:{mins:02d}:{sec:02d}"
                url = f"https://youtube.com/watch?v={
                    m['video_id']}&t={start_sec}s"
                print(f"  [{ts}] {s['text'].strip()}")
                print(f"  → {url}")
                print()


# ── 2. Find potential identity splits (same person, two global IDs) ────────────

def find_potential_splits(similarity_threshold: float = 0.82):
    """
    Compare all speaker embeddings in Qdrant against each other.
    If two different global_ids have high cosine similarity, they might
    be the same real person that TitaNet failed to merge.
    Raises threshold slightly above the match threshold to catch near-misses.
    """
    print(f"\nScanning for potential speaker identity splits "
          f"(similarity > {similarity_threshold})...")

    # Pull all speaker points from Qdrant
    results, offset = [], None
    while True:
        batch, offset = qdrant.scroll(
            collection_name="speakers",
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )
        results.extend(batch)
        if offset is None:
            break

    log.debug(f"Loaded {len(results)} speaker vectors from Qdrant")

    # Deduplicate to one vector per global_id (use first found)
    seen_globals = {}
    for r in results:
        gid = r.payload["global_id"]
        if gid not in seen_globals:
            seen_globals[gid] = {
                "global_id": gid,
                "display_name": r.payload["display_name"],
                "vector": r.vector
            }

    speakers = list(seen_globals.values())
    print(f"Comparing {len(speakers)} unique global speaker identities...\n")

    potential_splits = []
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            a = speakers[i]
            b = speakers[j]

            if a["global_id"] == b["global_id"]:
                continue

            # Cosine similarity
            import numpy as np
            va = np.array(a["vector"])
            vb = np.array(b["vector"])
            norm_a = np.linalg.norm(va)
            norm_b = np.linalg.norm(vb)
            if norm_a == 0 or norm_b == 0:
                continue
            sim = float(np.dot(va, vb) / (norm_a * norm_b))

            if sim >= similarity_threshold:
                potential_splits.append((sim, a, b))

    if not potential_splits:
        print("✓ No potential identity splits found above threshold.")
        return

    potential_splits.sort(reverse=True, key=lambda x: x[0])
    print(f"⚠ Found {len(potential_splits)} potential split(s):\n")
    print(f"  {'Similarity':>10}  {'Speaker A':<25} {'Speaker B'}")
    print(f"  {'-'*65}")
    for sim, a, b in potential_splits:
        print(f"  {sim:>10.4f}  {a['display_name']:<25} {b['display_name']}")

    print(f"\nTo merge two speakers, run:")
    print(f"  python verify_speakers.py merge <global_id_keep> <global_id_remove>")


# ── 3. Show sample transcript lines for a global speaker across VODs ──────────

def show_cross_vod_samples(global_id: str, samples_per_vod: int = 3):
    """
    Pull sample transcript lines for a global speaker from each VOD,
    with clickable YouTube timestamp URLs for manual verification.
    Sorted by VOD date ascending.
    """
    with db.get_conn() as conn:
        mappings = conn.execute("""
            SELECT sgm.vod_id, sgm.diarization_label,
                   v.title, v.date, v.video_id
            FROM speaker_global_map sgm
            JOIN global_speakers gs ON sgm.global_speaker_id = gs.id
            JOIN vods v ON sgm.vod_id = v.id
            WHERE gs.global_id = ?
            ORDER BY v.date ASC
        """, (global_id,)).fetchall()

        if not mappings:
            print(f"No mappings found for global_id={global_id}")
            return

        speaker_name = conn.execute(
            "SELECT display_name FROM global_speakers WHERE global_id = ?",
            (global_id,)
        ).fetchone()["display_name"]

    print(f"\nCross-VOD verification: {speaker_name} ({global_id})")
    print(f"Appears in {len(mappings)} VOD(s)\n")

    with db.get_conn() as conn:
        for m in mappings:
            print(f"{'='*65}")
            print(f"VOD {m['vod_id']}: {m['title']}")
            print(f"Date: {m['date']} | Label: {m['diarization_label']}")
            print(f"{'-'*65}")

            samples = conn.execute("""
                SELECT start_time, text
                FROM transcript_segments
                WHERE vod_id = ? AND diarization_label = ?
                AND LENGTH(TRIM(text)) > 40
                ORDER BY RANDOM()
                LIMIT ?
            """, (m["vod_id"], m["diarization_label"],
                  samples_per_vod)).fetchall()

            if not samples:
                print("  (no qualifying segments found)")
                continue

            for s in samples:
                start_sec = int(s["start_time"])
                h = start_sec // 3600
                mins = (start_sec % 3600) // 60
                sec = start_sec % 60
                ts = f"{h:02d}:{mins:02d}:{sec:02d}"
                url = f"https://youtube.com/watch?v={
                    m['video_id']}&t={start_sec}s"
                print(f"  [{ts}] {s['text'].strip()}")
                print(f"  → {url}")
                print()

# ── 4. Merge two global speakers into one ─────────────────────────────────────


def merge_speakers(keep_global_id: str, remove_global_id: str):
    """
    Merge remove_global_id into keep_global_id.
    Updates speaker_global_map, global_speakers, and Qdrant payloads.
    """
    with db.get_conn() as conn:
        keep = conn.execute(
            "SELECT * FROM global_speakers WHERE global_id = ?",
            (keep_global_id,)
        ).fetchone()
        remove = conn.execute(
            "SELECT * FROM global_speakers WHERE global_id = ?",
            (remove_global_id,)
        ).fetchone()

    if not keep:
        print(f"✗ global_id '{keep_global_id}' not found")
        return
    if not remove:
        print(f"✗ global_id '{remove_global_id}' not found")
        return

    print(f"\nMerging:")
    print(f"  REMOVE: {remove['display_name']} ({remove_global_id})")
    print(f"  → INTO: {keep['display_name']} ({keep_global_id})")
    confirm = input("\nConfirm merge? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    with db.get_conn() as conn:
        # Remap all speaker_global_map entries
        conn.execute("""
            UPDATE speaker_global_map
            SET global_speaker_id = (
                SELECT id FROM global_speakers WHERE global_id = ?
            )
            WHERE global_speaker_id = (
                SELECT id FROM global_speakers WHERE global_id = ?
            )
        """, (keep_global_id, remove_global_id))

        # Delete the removed global speaker
        conn.execute(
            "DELETE FROM global_speakers WHERE global_id = ?",
            (remove_global_id,)
        )

    log.info(f"SQLite merge complete: {remove_global_id} → {keep_global_id}")

    # Update Qdrant payloads for the removed global_id
    results, offset = [], None
    while True:
        batch, offset = qdrant.scroll(
            collection_name="speakers",
            scroll_filter=Filter(must=[
                FieldCondition(key="global_id",
                               match=MatchValue(value=remove_global_id))
            ]),
            limit=100, offset=offset, with_payload=True
        )
        results.extend(batch)
        if offset is None:
            break

    for r in results:
        qdrant.set_payload("speakers",
                           payload={"global_id": keep_global_id,
                                    "display_name": keep["display_name"]},
                           points=[r.id])

    log.info(f"Updated {len(results)} Qdrant speaker point(s)")
    print(f"\n✓ Merged {len(results)} Qdrant points")
    print(f"✓ All '{remove['display_name']}' entries now attributed "
          f"to '{keep['display_name']}'")
    print(f"\n⚠ Re-ingest affected VODs to update transcript chunks:")
    print(f"  sqlite3 /data/streams/streams.db \"DELETE FROM ingest_log;\"")
    print(f"  python ingest.py")


def show_titanet_rank(top: int = None):
    """
    Rank global speakers by total segment count across ALL vods.
    The main streamer should be #1 by a large margin.
    Shows consistency score — what % of their vods they appear in.
    """
    with db.get_conn() as conn:
        total_vods = conn.execute(
            "SELECT COUNT(*) FROM vods WHERE has_transcript = 1"
        ).fetchone()[0]

        total_segments = conn.execute(
            "SELECT COUNT(*) FROM transcript_segments "
            "WHERE diarization_label IS NOT NULL "
            "AND diarization_label != 'SPEAKER_UNKNOWN'"
        ).fetchone()[0]

        rows = conn.execute("""
            SELECT
                gs.global_id,
                gs.display_name,
                COUNT(ts.id) as total_segments,
                COUNT(DISTINCT sgm.vod_id) as vod_count,
                ROUND(COUNT(ts.id) * 100.0 / ?, 1) as pct_of_all_segs,
                ROUND(COUNT(DISTINCT sgm.vod_id) * 100.0 / ?, 1) as vod_coverage_pct
            FROM global_speakers gs
            JOIN speaker_global_map sgm ON sgm.global_speaker_id = gs.id
            JOIN transcript_segments ts
                ON ts.vod_id = sgm.vod_id
                AND ts.diarization_label = sgm.diarization_label
            GROUP BY gs.global_id
            ORDER BY total_segments DESC
        """, (total_segments, total_vods)).fetchall()

    if not rows:
        print("No global speakers found. Run ingest first.")
        return

    if top:
        rows = rows[:top]

    print(f"\nTitaNet Global Speaker Rank")
    print(f"Total VODs with transcripts : {total_vods}")
    print(f"Total labeled segments      : {total_segments:,}")
    print()
    print(f"{'Rank':<5} {'Display Name':<25} {'Segments':>10} "
          f"{'% Segs':>8} {'VODs':>6} {'% VODs':>8}  {'Identity'}")
    print("─" * 85)

    for rank, r in enumerate(rows, 1):
        named = r["display_name"]
        is_named = not named.startswith("spk_")
        name_display = named if is_named else "⚠ NOT NAMED"

        # Flag the likely main streamer
        flag = ""
        if rank == 1:
            flag = " ◄ likely main streamer"
        elif r["vod_coverage_pct"] >= 80:
            flag = " ◄ recurring"
        elif r["vod_coverage_pct"] <= 15:
            flag = " ◄ rare guest"

        print(f"{rank:<5} {name_display:<25} {r['total_segments']:>10,} "
              f"{r['pct_of_all_segs']:>7.1f}% {r['vod_count']:>6} "
              f"{r['vod_coverage_pct']:>7.1f}%  {r['global_id']}{flag}")

    print()
    print("% Segs  = share of all labeled transcript segments")
    print("% VODs  = % of total VODs this speaker appears in")
    print()
    print("To verify a speaker, run:")
    print("  python verify_speakers.py samples <global_id>")
    print("To rename a confirmed speaker, run:")
    print("  python speakers.py rename <global_id> \"Real Name\"")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "rank"

    if cmd == "rank":
        top = int(sys.argv[2]) if len(sys.argv) > 2 else None
        show_titanet_rank(top)

    elif cmd == "coverage":
        top = int(sys.argv[2]) if len(sys.argv) > 2 else None
        show_global_speaker_coverage(top)

    elif cmd == "splits":
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.82
        find_potential_splits(threshold)

    elif cmd == "samples" and len(sys.argv) >= 3:
        show_cross_vod_samples(sys.argv[2])

    elif cmd == "merge" and len(sys.argv) == 4:
        merge_speakers(sys.argv[2], sys.argv[3])

    else:
        print("Usage:")
        print("  python verify_speakers.py rank [top_n]")
        print("    → Rank all speakers by prevalence — find your main streamer")
        print("  python verify_speakers.py coverage [top_n]")
        print("    → Full per-VOD breakdown per speaker")
        print("  python verify_speakers.py splits [threshold]")
        print("    → Find speakers TitaNet may have failed to merge")
        print("  python verify_speakers.py samples <global_id>")
        print("    → Sample transcript lines + YouTube URLs for manual verification")
        print("  python verify_speakers.py merge <keep_id> <remove_id>")
        print("    → Merge two global speakers into one")
