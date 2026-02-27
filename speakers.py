import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from logger import get_logger
import db
import speaker_id as spkr

load_dotenv()
log = get_logger("speakers")
qdrant = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)


def list_speakers():
    log.debug("Listing global speakers")
    speakers = db.list_global_speakers()
    if not speakers:
        log.info("No global speakers registered yet")
        print("No global speakers registered yet.")
        return
    print(f"\n{'Global ID':<22} {'Display Name':<30} {'Created'}")
    print("─" * 72)
    for s in speakers:
        print(f"{s['global_id']:<22} {
              s['display_name']:<30} {s['created_at']}")
    log.info(f"Listed {len(speakers)} global speaker(s)")


def rename_speaker(global_id: str, new_name: str):
    log.info(f"Renaming speaker {global_id} → '{new_name}'")
    db.rename_global_speaker(global_id, new_name)
    spkr.update_speaker_display_name(global_id, new_name)
    print(f"✓ Renamed '{global_id}' → '{new_name}' in SQLite + Qdrant")
    log.info(f"Rename complete: {global_id} → '{new_name}'")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        list_speakers()
    elif cmd == "rename" and len(sys.argv) == 4:
        rename_speaker(sys.argv[2], sys.argv[3])
    else:
        print("Usage:")
        print("  python speakers.py list")
        print("  python speakers.py rename <global_id> <new_name>")
