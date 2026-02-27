import os
import uuid
import tempfile
import numpy as np
import subprocess
import torch
from pydub import AudioSegment
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
from logger import get_logger

load_dotenv()
log = get_logger("speaker_id")

AUDIO_DIR = os.getenv("AUDIO_DIR", "/data/streams/audio")
THRESHOLD = float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.75"))
qdrant = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)

_speaker_model = None


def get_speaker_model():
    global _speaker_model
    if _speaker_model is None:
        log.info(
            "Loading TitaNet speaker model (first load, this may take a moment)...")
        import nemo.collections.asr as nemo_asr
        _speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large")
        _speaker_model.eval()
        log.info("TitaNet model loaded successfully")
    return _speaker_model


def ensure_speaker_collection():
    existing = {c.name for c in qdrant.get_collections().collections}
    if "speakers" not in existing:
        log.info("Creating Qdrant 'speakers' collection")
        qdrant.create_collection("speakers",
                                 vectors_config=VectorParams(size=192, distance=Distance.COSINE))
        log.info("'speakers' collection created")
    else:
        log.debug("'speakers' collection already exists")


def get_voice_embedding(audio_segment: AudioSegment) -> np.ndarray:
    log.debug(f"Generating voice embedding for {
              len(audio_segment)}ms audio clip")
    model = get_speaker_model()
    clip = audio_segment.set_frame_rate(16000).set_channels(1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        clip.export(f.name, format="wav")
        tmp_path = f.name
    try:
        with torch.no_grad():
            emb = model.get_embedding(tmp_path)
        log.debug("Voice embedding generated successfully")
    except Exception as e:
        log.error(f"Failed to generate voice embedding: {e}")
        raise
    finally:
        os.remove(tmp_path)
        log.debug(f"Cleaned up temp file {tmp_path}")
    return (emb.squeeze().cpu().numpy()
            if isinstance(emb, torch.Tensor) else np.array(emb).squeeze())


def resolve_speaker(audio_clip: AudioSegment) -> tuple[str, str, float]:
    log.debug(f"Resolving speaker for {
              len(audio_clip)}ms clip (threshold={THRESHOLD})")
    ensure_speaker_collection()
    emb = get_voice_embedding(audio_clip)

    results = qdrant.query_points(
        collection_name="speakers",
        query=emb.tolist(),
        limit=1,
        score_threshold=THRESHOLD
    ).points

    if results:
        p = results[0].payload
        log.info(f"Matched existing speaker: '{p['display_name']}' "
                 f"({p['global_id']}) score={results[0].score:.4f}")
        return p["global_id"], p["display_name"], results[0].score

    global_id = f"spk_{uuid.uuid4().hex[:8]}"
    log.info(
        f"No match found above threshold — registering new speaker: {global_id}")
    qdrant.upsert("speakers", points=[PointStruct(
        id=uuid.uuid4().hex,
        vector=emb.tolist(),
        payload={"global_id": global_id, "display_name": global_id}
    )])
    log.debug(f"New speaker {global_id} stored in Qdrant")
    return global_id, global_id, 1.0


def get_speaker_clips_from_vod_by_label(filename: str, segments: list,
                                        diarization_label: str,
                                        max_clips: int = 5,
                                        clip_duration_sec: int = 10,
                                        min_text_words: int = 8) -> list[AudioSegment]:
    """
    Extract audio clips for a speaker label, preferring longer segments
    with more words (more likely to be clean speech vs. reactions/crosstalk).
    """
    audio_path = os.path.join(AUDIO_DIR, filename)
    log.debug(f"Extracting clips for '{diarization_label}' from {audio_path}")

    if not os.path.exists(audio_path):
        log.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    # Filter to segments with enough words to be clean speech
    speaker_segs = [
        s for s in segments
        if s["diarization_label"] == diarization_label
        and s["text"] and len(s["text"].split()) >= min_text_words
    ]

    if not speaker_segs:
        # Fall back to any segment if word filter is too strict
        speaker_segs = [
            s for s in segments
            if s["diarization_label"] == diarization_label
        ]
        log.debug(f"Word filter too strict for '{diarization_label}', "
                  f"using all {len(speaker_segs)} segments")

    if not speaker_segs:
        return []

    # Sort by text length descending — longer speech = cleaner embedding
    speaker_segs_sorted = sorted(
        speaker_segs,
        key=lambda s: len(s["text"]),
        reverse=True
    )

    # Take top candidates but spread them across the file for diversity
    # Split file into thirds and pick the longest segment from each third
    if len(speaker_segs) >= 3:
        total_duration = speaker_segs[-1]["start_time"] - \
            speaker_segs[0]["start_time"]
        third = total_duration / 3
        start_time = speaker_segs[0]["start_time"]

        thirds = [[], [], []]
        for s in speaker_segs:
            offset = s["start_time"] - start_time
            idx = min(int(offset / third), 2) if total_duration > 0 else 0
            thirds[idx].append(s)

        selected = []
        for third_segs in thirds:
            if third_segs:
                # Pick longest from each third
                best = max(third_segs, key=lambda s: len(s["text"]))
                selected.append(best)

        # Fill remaining slots from top of sorted list if needed
        existing_ids = {id(s) for s in selected}
        for s in speaker_segs_sorted:
            if len(selected) >= max_clips:
                break
            if id(s) not in existing_ids:
                selected.append(s)
    else:
        selected = speaker_segs_sorted[:max_clips]

    log.debug(f"Selected {len(selected)} candidate segments for '{
              diarization_label}'")

    clips = []
    for seg in selected:
        start_sec = seg["start_time"]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", audio_path,
            "-t", str(clip_duration_sec),
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            tmp_path,
            "-loglevel", "error"
        ]

        try:
            import subprocess
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode != 0:
                log.warning(f"ffmpeg failed at {start_sec}s: "
                            f"{result.stderr.decode()}")
                os.remove(tmp_path)
                continue

            clip = AudioSegment.from_wav(tmp_path)
            if len(clip) >= 2000:
                clips.append(clip)
                log.debug(f"  Clip at {start_sec}s: "
                          f"'{seg['text'][:50]}...' ({len(clip)}ms)")
            else:
                log.debug(f"  Skipped short clip at {start_sec}s")

        except subprocess.TimeoutExpired:
            log.error(f"ffmpeg timed out at {start_sec}s")
        except Exception as e:
            log.error(f"Clip extraction failed at {start_sec}s: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    log.info(f"Extracted {len(clips)} clip(s) for '{diarization_label}'")
    return clips


def update_speaker_display_name(global_id: str, new_name: str):
    log.info(f"Updating Qdrant display_name for {global_id} → '{new_name}'")
    results = qdrant.scroll(
        collection_name="speakers",
        scroll_filter=Filter(must=[
            FieldCondition(key="global_id", match=MatchValue(value=global_id))
        ]),
        limit=100,
        with_payload=True
    )[0]
    log.debug(f"Found {len(results)} Qdrant points for global_id={global_id}")
    for r in results:
        qdrant.set_payload("speakers",
                           payload={"display_name": new_name}, points=[r.id])
    log.info(f"Updated {len(results)} Qdrant point(s) for {global_id}")
