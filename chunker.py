from logger import get_logger

log = get_logger("chunker")

MAX_TOKENS_PER_CHUNK = 512   # hard cap per chunk
OVERLAP_TOKENS = 75          # ~15% overlap between consecutive chunks
# Rough token estimator: 1 token ≈ 4 chars for English transcript text
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def build_line(seg, display_name: str, sec_to_ts) -> str:
    return f"[{sec_to_ts(seg['start_time'])}] {display_name}: {seg['text'].strip()}"


def speaker_turn_chunks(segments: list, speaker_map: dict,
                        sec_to_ts, vod: dict) -> list[dict]:
    """
    Phase 1: Split on speaker turns first. Each contiguous block of
    segments from the same speaker becomes a candidate chunk.
    Then apply a token-size cap so long monologues get sub-split
    with overlap.
    """
    if not segments:
        return []

    log.info(f"Chunking {len(segments)} segments using speaker-turn strategy")

    def yt_link(video_id, start_sec):
        return f"https://youtube.com/watch?v={video_id}&t={int(start_sec)}s"

    def make_chunk(lines, speakers, start_sec, end_sec):
        return {
            "text": "\n".join(lines),
            "vod_id": vod["id"],
            "video_id": vod["video_id"],
            "title": vod["title"],
            "date": vod["date"],
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_ts": sec_to_ts(start_sec),
            "youtube_url": yt_link(vod["video_id"], start_sec),
            "speakers": list(speakers),
            "chunk_type": "speaker_turn"
        }

    # ── Step 1: Group into speaker turns ─────────────────────────────────────
    turns = []
    current_turn = []
    current_label = None

    for seg in segments:
        label = seg["diarization_label"] or "SPEAKER_UNKNOWN"
        if label != current_label:
            if current_turn:
                turns.append((current_label, current_turn))
            current_turn = [seg]
            current_label = label
        else:
            current_turn.append(seg)

    if current_turn:
        turns.append((current_label, current_turn))

    log.info(f"Found {len(turns)} speaker turns")

    # ── Step 2: Apply token cap with overlap to each turn ────────────────────
    chunks = []

    for label, turn_segs in turns:
        global_id, display_name = speaker_map.get(label, (label, label))

        # Build all lines for this turn
        lines = [build_line(s, display_name, sec_to_ts) for s in turn_segs]
        total_tokens = estimate_tokens("\n".join(lines))

        if total_tokens <= MAX_TOKENS_PER_CHUNK:
            # Entire turn fits in one chunk
            chunks.append(make_chunk(
                lines,
                {global_id},
                turn_segs[0]["start_time"],
                turn_segs[-1]["start_time"]
            ))
            log.debug(f"  Turn '{label}' → 1 chunk ({total_tokens} tokens)")
        else:
            # Turn is too long — sub-split with overlap
            log.debug(f"  Turn '{label}' is {
                      total_tokens} tokens — sub-splitting")
            sub_chunks = split_with_overlap(
                lines, turn_segs, global_id, display_name,
                vod, sec_to_ts, yt_link
            )
            chunks.extend(sub_chunks)
            log.debug(f"  Turn '{label}' → {len(sub_chunks)} sub-chunks")

    log.info(f"Speaker-turn chunking complete: {len(chunks)} total chunks")
    return chunks


def split_with_overlap(lines: list, segs: list, global_id: str,
                       display_name: str, vod: dict,
                       sec_to_ts, yt_link) -> list[dict]:
    """
    Sub-split a long speaker turn into token-capped chunks with overlap.
    Overlap is implemented by including the last N lines of the previous
    chunk at the start of the next one.
    """
    chunks = []
    i = 0
    overlap_lines = []

    while i < len(lines):
        current_lines = list(overlap_lines)  # start with overlap from previous
        current_start = segs[i]["start_time"] if not overlap_lines else \
            segs[max(0, i - len(overlap_lines))]["start_time"]
        chunk_start_idx = i

        # Fill chunk up to token cap
        while i < len(lines):
            candidate = current_lines + [lines[i]]
            if estimate_tokens("\n".join(candidate)) > MAX_TOKENS_PER_CHUNK:
                break
            current_lines.append(lines[i])
            i += 1

        # If we didn't advance at all (single line > MAX_TOKENS), force include it
        if i == chunk_start_idx:
            current_lines.append(lines[i])
            i += 1

        end_sec = segs[min(i, len(segs) - 1)]["start_time"]

        chunks.append({
            "text": "\n".join(current_lines),
            "vod_id": vod["id"],
            "video_id": vod["video_id"],
            "title": vod["title"],
            "date": vod["date"],
            "start_sec": current_start,
            "end_sec": end_sec,
            "start_ts": sec_to_ts(current_start),
            "youtube_url": yt_link(vod["video_id"], current_start),
            "speakers": [global_id],
            "chunk_type": "speaker_turn_sub"
        })

        # Carry last OVERLAP_TOKENS worth of lines into next chunk
        overlap_lines = []
        overlap_token_count = 0
        for line in reversed(current_lines):
            line_tokens = estimate_tokens(line)
            if overlap_token_count + line_tokens > OVERLAP_TOKENS:
                break
            overlap_lines.insert(0, line)
            overlap_token_count += line_tokens

    return chunks
