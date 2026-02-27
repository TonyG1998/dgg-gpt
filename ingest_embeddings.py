import os
import uuid
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
from logger import get_logger
import db
from ingest import sec_to_ts, yt_link
from chunker import speaker_turn_chunks

load_dotenv()
log = get_logger("ingest_embeddings")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)


def ensure_transcript_collection():
    existing = {c.name for c in qdrant.get_collections().collections}
    if "transcripts" not in existing:
        log.info("Creating Qdrant 'transcripts' collection")
        qdrant.create_collection("transcripts",
                                 vectors_config=VectorParams(size=1536, distance=Distance.COSINE))


def embed_text(text: str) -> list[float]:
    r = openai_client.embeddings.create(
        model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        input=text
    )
    return r.data[0].embedding


def embed_vod(vod_id: int):
    ensure_transcript_collection()

    vod = db.get_vod_by_id(vod_id)
    if not vod:
        log.error(f"VOD {vod_id} not found")
        return 0

    # Require speaker mappings to exist before embedding
    speaker_map = db.get_global_speaker_map_for_vod(vod_id)
    if not speaker_map:
        log.warning(f"VOD {vod_id} has no speaker mappings yet — "
                    f"run ingest_speakers.py first")
        return 0

    segments = db.get_segments_for_vod(vod_id)
    if not segments:
        log.warning(f"No segments for VOD {vod_id}")
        return 0

    log.info(f"{'='*60}")
    log.info(f"Embedding VOD {vod_id}: '{vod['title']}'")
    log.info(f"  {len(segments)} segments, {
             len(speaker_map)} speaker mappings")

    chunks = speaker_turn_chunks(segments, speaker_map, sec_to_ts, dict(vod))
    log.info(f"  Created {len(chunks)} chunks")

    points = []
    failed = 0
    for i, chunk in enumerate(chunks):
        try:
            emb = embed_text(chunk["text"])
            points.append(PointStruct(
                id=uuid.uuid4().hex,
                vector=emb,
                payload=chunk
            ))
        except Exception as e:
            log.error(f"  Embed failed chunk {i+1}: {e}")
            failed += 1

        if (i + 1) % 20 == 0:
            log.info(f"  Embedded {i+1}/{len(chunks)} chunks...")

    if failed:
        log.warning(f"  {failed} chunk(s) failed")

    qdrant.upsert("transcripts", points=points)
    db.mark_vod_ingested(vod_id, len(points))
    log.info(f"✓ VOD {vod_id} embedded — {len(points)} chunks stored")
    return len(points)


def embed_all_pending():
    """
    Embed all VODs that have speaker mappings but haven't been
    embedded yet. This is the only step that calls OpenAI.
    """
    with db.get_conn() as conn:
        vods = conn.execute("""
            SELECT v.* FROM vods v
            WHERE v.has_transcript = 1
            AND v.id NOT IN (SELECT vod_id FROM ingest_log)
            AND v.id IN (SELECT DISTINCT vod_id FROM speaker_global_map)
            ORDER BY v.date ASC
        """).fetchall()

    if not vods:
        log.info("No VODs pending embedding")
        return

    log.info(
        f"Found {len(vods)} VOD(s) ready to embed (have speaker mappings)")
    total = 0
    for i, vod in enumerate(vods, 1):
        log.info(f"Progress: {i}/{len(vods)} VODs")
        try:
            total += embed_vod(vod["id"])
        except Exception as e:
            log.error(f"Failed for VOD {vod['id']}: {e} — continuing")
            continue

    log.info(f"Embedding complete — {total} total chunks stored")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        embed_vod(int(sys.argv[1]))
    else:
        embed_all_pending()
