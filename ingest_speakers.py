import os
from dotenv import load_dotenv
from logger import get_logger
import db
import speaker_id as spkr
import uuid
import numpy as np
from qdrant_client.models import ScoredPoint

load_dotenv()
log = get_logger("ingest_speakers")


def resolve_speakers_for_vod(vod_id: int):
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    qdrant = QdrantClient(host=os.getenv(
        "QDRANT_HOST", "localhost"), port=6333)
    spkr.ensure_speaker_collection()

    vod = db.get_vod_by_id(vod_id)
    if not vod:
        log.error(f"VOD {vod_id} not found")
        return

    # Skip if already done
    with db.get_conn() as conn:
        already_mapped = conn.execute("""
            SELECT COUNT(*) FROM speaker_global_map WHERE vod_id = ?
        """, (vod_id,)).fetchone()[0]
    if already_mapped:
        log.info(f"VOD {vod_id} already has {
                 already_mapped} speaker mappings — skipping")
        return "skipped"

    log.info(f"{'='*60}")
    log.info(f"Resolving speakers for VOD {vod_id}: '{vod['title']}'")

    segments = db.get_segments_for_vod(vod_id)
    if not segments:
        log.warning(f"No segments for VOD {vod_id} — skipping")
        return

    unique_labels = db.get_unique_speakers_for_vod(vod_id)
    log.info(f"  Labels to resolve: "
             f"{[r['diarization_label'] for r in unique_labels]}")

    audio_path = os.path.join(
        os.getenv("AUDIO_DIR", "/data/streams/audio"), vod["filename"])
    audio_available = os.path.exists(audio_path)
    log.info(f"  Audio: {'found' if audio_available else 'NOT FOUND'}")

    for row in unique_labels:
        label = row["diarization_label"]
        log.info(f"  Processing '{label}'...")
        global_id = None
        display_name = None
        score = 0.0

        if audio_available:
            try:
                clips = spkr.get_speaker_clips_from_vod_by_label(
                    vod["filename"], segments, label)
                if not clips:
                    raise ValueError("No clips extracted")

                from speaker_id import get_voice_embedding
                embeddings = [get_voice_embedding(clip) for clip in clips]
                emb = np.mean(embeddings, axis=0)
                # normalize for cosine similarity
                emb = emb / np.linalg.norm(emb)

                results = qdrant.query_points(
                    collection_name="speakers",
                    query=emb.tolist(),
                    limit=1,
                    score_threshold=float(
                        os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.55"))
                ).points
                if results:
                    global_id = results[0].payload["global_id"]
                    display_name = results[0].payload["display_name"]
                    score = results[0].score
                    log.info(f"    Matched '{display_name}' ({global_id}) "
                             f"score={score:.4f}")
                else:
                    global_id = f"spk_{uuid.uuid4().hex[:8]}"
                    display_name = global_id
                    score = 1.0
                    log.info(f"    New speaker → {global_id}")
                    qdrant.upsert("speakers", points=[PointStruct(
                        id=uuid.uuid4().hex,
                        vector=emb.tolist(),
                        payload={"global_id": global_id,
                                 "display_name": display_name,
                                 "is_enrolled": False}
                    )])

            except Exception as e:
                log.warning(f"    TitaNet failed for '{
                            label}': {e} — placeholder")
                global_id = f"spk_{uuid.uuid4().hex[:8]}"
                display_name = global_id
                score = 0.0
        else:
            global_id = f"spk_{uuid.uuid4().hex[:8]}"
            display_name = global_id
            score = 0.0
            log.info(f"    No audio — placeholder {global_id}")

        gs_db_id = db.upsert_global_speaker(global_id, display_name)
        db.save_speaker_mapping(label, vod_id, gs_db_id, score)
        log.info(f"    Saved: '{label}' → '{display_name}' ({global_id})")

    with db.get_conn() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO speaker_ingest_log (vod_id, speaker_count)
            VALUES (?, ?)
        """, (vod_id, len(unique_labels)))
    log.info(f"✓ Speaker resolution complete for VOD {vod_id}")


def resolve_all_pending(limit: int = None):
    with db.get_conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT v.id, v.title
            FROM vods v
            WHERE v.id NOT IN (SELECT vod_id FROM speaker_ingest_log)
            ORDER BY v.id DESC
        """).fetchall()

    total_available = len(rows)
    if limit:
        rows = rows[:limit]

    total = len(rows)
    if total == 0:
        log.info("No pending VODs found — all speakers resolved.")
        return

    log.info(
        f"Found {total_available} pending VOD(s) total, processing {total}")
    log.info("=" * 60)

    resolved = 0
    skipped = 0
    failed = 0

    for i, row in enumerate(rows, start=1):
        vod_id, title = row["id"], row["title"]
        short_title = title[:60] + \
            "..." if title and len(title) > 60 else title
        log.info(f"[{i}/{total}] VOD {vod_id}: {short_title}")

        try:
            result = resolve_speakers_for_vod(vod_id)
            if result == "skipped":
                skipped += 1
                log.info(f"[{i}/{total}] ↩ Skipped (already done)")
            else:
                resolved += 1
                log.info(
                    f"[{i}/{total}] ✓ Done — {resolved} resolved, {skipped} skipped, {failed} failed so far")
        except Exception as e:
            failed += 1
            log.error(f"[{i}/{total}] ✗ Failed VOD {vod_id}: {e}")

        if i % 10 == 0 or i == total:
            pct = int((i / total) * 100)
            bar = ("█" * (pct // 5)).ljust(20)
            log.info(f"  Progress: |{bar}| {
                     pct}% ({i}/{total}) — ✓{resolved} ↩{skipped} ✗{failed}")

    log.info("=" * 60)
    log.info(f"Ingest complete: {resolved} resolved, {
             skipped} skipped, {failed} failed out of {total} VOD(s)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("vod_id", nargs="?", type=int,
                        help="Single VOD ID to process")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process this many VODs (for testing)")
    args = parser.parse_args()

    if args.vod_id:
        resolve_speakers_for_vod(args.vod_id)
    else:
        resolve_all_pending(limit=args.limit)
