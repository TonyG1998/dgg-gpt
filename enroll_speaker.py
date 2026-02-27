import os
import sys
from pydub import AudioSegment
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from logger import get_logger
import speaker_id as spkr
import db

load_dotenv()
log = get_logger("enroll")
qdrant = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)


def enroll(audio_path: str, display_name: str, global_id: str = None):
    """
    Enroll a known speaker from a clean audio sample.
    This creates a high-quality reference embedding that all future
    diarized segments will be compared against.
    """
# Check if already enrolled and delete existing points first
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    existing, _ = qdrant.scroll(
        "speakers",
        scroll_filter=Filter(must=[
            FieldCondition(key="global_id", match=MatchValue(value=global_id))
        ]),
        limit=100,
        with_payload=False
    )
    if existing:
        qdrant.delete("speakers", points_selector=[p.id for p in existing])
        log.info(f"Removed {len(existing)} existing point(s) for {
                 global_id} before re-enrolling")

    if not global_id:
        import uuid
        global_id = f"spk_{uuid.uuid4().hex[:8]}"

    log.info(f"Enrolling speaker '{
             display_name}' ({global_id}) from {audio_path}")
    spkr.ensure_speaker_collection()

    # Load audio and split into 3 chunks for a robust averaged embedding
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    duration_ms = len(audio)
    log.info(f"Audio loaded: {duration_ms}ms ({duration_ms/1000:.1f}s)")

    # Split into 10-second chunks spread across the sample
    chunk_ms = 10000
    num_chunks = min(5, duration_ms // chunk_ms)
    if num_chunks < 1:
        log.error("Audio too short — need at least 10 seconds")
        return

    step = duration_ms // num_chunks
    clips = [audio[i*step: i*step + chunk_ms] for i in range(num_chunks)]

    import numpy as np
    embs = [spkr.get_voice_embedding(clip) for clip in clips]
    avg_emb = np.mean(embs, axis=0)
    norm = np.linalg.norm(avg_emb)
    if norm > 0:
        avg_emb = avg_emb / norm

    # Store in Qdrant with is_enrolled flag
    import uuid as uuid_mod
    qdrant.upsert("speakers", points=[PointStruct(
        id=uuid_mod.uuid4().hex,
        vector=avg_emb.tolist(),
        payload={
            "global_id": global_id,
            "display_name": display_name,
            "is_enrolled": True  # marks this as a known-speaker reference
        }
    )])

    # Register in SQLite
    db.upsert_global_speaker(global_id, display_name)

    log.info(f"✓ Enrolled '{display_name}' ({
             global_id}) from {num_chunks} clips")
    print(f"✓ Enrolled '{display_name}' as {global_id}")
    print(f"  Source: {audio_path}")
    print(f"  Chunks averaged: {num_chunks}")
    print(f"  This speaker will now be matched automatically during ingest")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python enroll_speaker.py <audio_file> \"Display Name\" [global_id]")
        print("")
        print("Examples:")
        print("  python enroll_speaker.py ~/clips/destiny_solo.wav \"Destiny\"")
        print("  python enroll_speaker.py ~/clips/dan.wav \"Dan\" spk_dan")
        sys.exit(1)

    audio_file = sys.argv[1]
    name = sys.argv[2]
    gid = sys.argv[3] if len(sys.argv) > 3 else None
    enroll(audio_file, name, gid)
