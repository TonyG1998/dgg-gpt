import os
import uuid
from dotenv import load_dotenv
from logger import get_logger
import db

load_dotenv()
log = get_logger("map_speakers")


def show_speaker_samples(vod_id: int, num_samples: int = 3):
    log.debug(f"Fetching speaker samples for vod_id={vod_id}")
    segments = db.get_segments_for_vod(vod_id)
    vod = db.get_vod_by_id(vod_id)

    if not vod:
        log.error(f"VOD {vod_id} not found")
        return

    log.info(f"Showing samples for vod_id={vod_id} '{vod['title']}'")
    print(f"\nVOD {vod_id}: {vod['title']} ({vod['date']})")
    print("=" * 60)

    seen = {}
    for seg in segments:
        label = seg["diarization_label"] or "SPEAKER_UNKNOWN"
        if label not in seen:
            seen[label] = []
        if len(seen[label]) < num_samples:
            seen[label].append(f"  [{int(seg['start_time'])}s] {
                               seg['text'].strip()}")

    log.debug(f"Found {len(seen)} unique labels")
    for label, lines in seen.items():
        print(f"\n{label}:")
        for line in lines:
            print(line)


def assign_name(vod_id: int, diarization_label: str, real_name: str):
    log.info(f"Assigning name: VOD {vod_id} | '{
             diarization_label}' → '{real_name}'")

    with db.get_conn() as conn:
        exists = conn.execute("""
            SELECT COUNT(*) FROM transcript_segments
            WHERE vod_id = ? AND diarization_label = ?
        """, (vod_id, diarization_label)).fetchone()[0]

    if not exists:
        log.error(f"No segments found for label '{
                  diarization_label}' in VOD {vod_id}")
        print(f"✗ No segments found for '{diarization_label}' in VOD {vod_id}")
        return

    log.debug(f"Found segments for '{diarization_label}' in VOD {vod_id}")

    with db.get_conn() as conn:
        existing_global = conn.execute(
            "SELECT * FROM global_speakers WHERE display_name = ?", (real_name,)
        ).fetchone()

    if existing_global:
        global_id = existing_global["global_id"]
        gs_db_id = existing_global["id"]
        log.info(f"Reusing existing global speaker '{
                 real_name}' ({global_id})")
        print(f"  Reusing existing global speaker '{real_name}' ({global_id})")
    else:
        global_id = f"spk_{uuid.uuid4().hex[:8]}"
        gs_db_id = db.upsert_global_speaker(global_id, real_name)
        log.info(f"Created new global speaker '{real_name}' ({global_id})")
        print(f"  Created new global speaker '{real_name}' ({global_id})")

    db.save_speaker_mapping(diarization_label, vod_id, gs_db_id, 1.0)
    log.info(f"✓ Mapping saved: VOD {vod_id} | '{
             diarization_label}' → '{real_name}'")
    print(f"✓ VOD {vod_id} | '{diarization_label}' → '{real_name}'")


def rank_speakers(vod_id: int = None):
    """
    Rank all diarization labels by number of segments, most to least.
    If vod_id is given, rank within that VOD only.
    Also shows whether each label has been named yet.
    """
    log.debug(f"Ranking speakers {'for vod_id=' +
              str(vod_id) if vod_id else 'across all vods'}")

    with db.get_conn() as conn:
        if vod_id:
            rows = conn.execute("""
                SELECT
                    ts.diarization_label,
                    COUNT(*) as segment_count,
                    COUNT(DISTINCT ts.vod_id) as vod_count,
                    ROUND(COUNT(*) * 100.0 / (
                        SELECT COUNT(*) FROM transcript_segments WHERE vod_id = ?
                    ), 1) as pct,
                    gs.display_name
                FROM transcript_segments ts
                LEFT JOIN speaker_global_map sgm
                    ON ts.diarization_label = sgm.diarization_label
                    AND ts.vod_id = sgm.vod_id
                LEFT JOIN global_speakers gs
                    ON sgm.global_speaker_id = gs.id
                WHERE ts.vod_id = ?
                AND ts.diarization_label IS NOT NULL
                AND ts.diarization_label != 'SPEAKER_UNKNOWN'
                GROUP BY ts.diarization_label
                ORDER BY segment_count DESC
            """, (vod_id, vod_id)).fetchall()
        else:
            rows = conn.execute("""
                SELECT
                    ts.diarization_label,
                    COUNT(*) as segment_count,
                    COUNT(DISTINCT ts.vod_id) as vod_count,
                    ROUND(COUNT(*) * 100.0 / (
                        SELECT COUNT(*) FROM transcript_segments
                        WHERE diarization_label IS NOT NULL
                    ), 1) as pct,
                    MAX(gs.display_name) as display_name
                FROM transcript_segments ts
                LEFT JOIN speaker_global_map sgm
                    ON ts.diarization_label = sgm.diarization_label
                    AND ts.vod_id = sgm.vod_id
                LEFT JOIN global_speakers gs
                    ON sgm.global_speaker_id = gs.id
                WHERE ts.diarization_label IS NOT NULL
                AND ts.diarization_label != 'SPEAKER_UNKNOWN'
                GROUP BY ts.diarization_label
                ORDER BY segment_count DESC
            """).fetchall()

    scope = f"VOD {vod_id}" if vod_id else "All VODs"
    print(f"\nSpeaker Rankings — {scope}")
    print(f"{'Rank':<6} {'Label':<14} {'Segments':>10} {
          'VODs':>6} {'% Total':>9} {'Named As'}")
    print("─" * 65)
    for i, r in enumerate(rows, 1):
        named = r["display_name"] or "⚠ NOT NAMED"
        print(f"{i:<6} {r['diarization_label']:<14} {r['segment_count']:>10,} "
              f"{r['vod_count']:>6} {r['pct']:>8.1f}% {named}")

    unnamed = sum(1 for r in rows if not r["display_name"])
    log.info(f"Ranked {len(rows)} labels — {unnamed} still unnamed")

    # Update __main__ to include rank command:
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python map_speakers.py rank                    # all vods")
        print("  python map_speakers.py rank <vod_id>           # specific vod")
        print("  python map_speakers.py samples <vod_id>")
        print("  python map_speakers.py assign <vod_id> <diarization_label> \"Real Name\"")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "rank":
        vod_id = int(sys.argv[2]) if len(sys.argv) == 3 else None
        rank_speakers(vod_id)

    elif cmd == "samples" and len(sys.argv) == 3:
        show_speaker_samples(int(sys.argv[2]))

    elif cmd == "assign" and len(sys.argv) == 5:
        assign_name(int(sys.argv[2]), sys.argv[3], sys.argv[4])

    else:
        print("Invalid arguments.")
