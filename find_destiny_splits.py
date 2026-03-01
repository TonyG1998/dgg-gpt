# find_destiny_splits.py
import os
import numpy as np
import sqlite3
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv()
qdrant = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)
DB_PATH = os.getenv("SQLITE_DB", "/data/streams/streams.db")

DESTINY_GLOBAL_ID = "spk_c591b7c2"
THRESHOLD = 0.40

# Get Destiny's vector
destiny_points, _ = qdrant.scroll(
    collection_name="speakers",
    scroll_filter=Filter(must=[
        FieldCondition(key="global_id", match=MatchValue(value=DESTINY_GLOBAL_ID))
    ]),
    limit=1,
    with_vectors=True,
    with_payload=True
)

if not destiny_points:
    print("Destiny not found in Qdrant")
    exit()

destiny_vec = np.array(destiny_points[0].vector)

# Query Qdrant for similar speakers
similar = qdrant.query_points(
    collection_name="speakers",
    query=destiny_vec.tolist(),
    limit=100,
    score_threshold=THRESHOLD
).points

# Filter out Destiny herself, collect global_ids
suspect_ids = [
    (r.payload["global_id"], r.payload["display_name"], r.score)
    for r in similar
    if r.payload["global_id"] != DESTINY_GLOBAL_ID
]

if not suspect_ids:
    print("No suspects found.")
    exit()

# Look up segment counts + VOD counts from SQLite (single query)
suspect_map = {gid: (name, score) for gid, name, score in suspect_ids}
id_list = list(suspect_map.keys())

placeholders = ",".join("?" * len(id_list))
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

rows = conn.execute(f"""
    SELECT
        gs.global_id,
        COUNT(ts.id) as total_segments,
        COUNT(DISTINCT sgm.vod_id) as total_vods
    FROM global_speakers gs
    JOIN speaker_global_map sgm ON sgm.global_speaker_id = gs.id
    JOIN transcript_segments ts
        ON ts.vod_id = sgm.vod_id
        AND ts.diarization_label = sgm.diarization_label
    WHERE gs.global_id IN ({placeholders})
    GROUP BY gs.global_id
""", id_list).fetchall()
conn.close()

results = []
for row in rows:
    gid = row["global_id"]
    name, score = suspect_map[gid]
    results.append({
        "global_id": gid,
        "display_name": name,
        "score": score,
        "segments": row["total_segments"],
        "vods": row["total_vods"],
    })

# Sort by segment count descending
results.sort(key=lambda x: x["segments"], reverse=True)

print(f"\nSuspect Destiny splits (similarity ≥ {THRESHOLD}), ranked by segments:\n")
print(f"  {'Score':>8}  {'Segments':>10}  {'VODs':>6}  {'global_id':<20}  display_name")
print(f"  {'-'*70}")
for r in results:
    print(f"  {r['score']:>8.4f}  {r['segments']:>10,}  {r['vods']:>6}  "
          f"{r['global_id']:<20}  {r['display_name']}")

print(f"\nTo sample a suspect:")
print(f"  python verify_speakers.py samples <global_id>")
print(f"To merge into Destiny:")
print(f"  python verify_speakers.py merge {DESTINY_GLOBAL_ID} <global_id>")

