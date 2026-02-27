import db


def print_status():
    rows = db.get_ingest_status()
    print(f"\n{'ID':<5} {'Date':<12} {'Title':<40} {
          'Transcript':<12} {'Ingested At':<22} {'Chunks'}")
    print("─" * 100)
    for r in rows:
        ingested = r["ingested_at"] if r["ingested_at"] else "NOT INGESTED"
        chunks = str(r["chunk_count"]) if r["chunk_count"] else "-"
        has_t = "✓" if r["has_transcript"] else "✗"
        title = (r["title"][:37] + "...") if len(r["title"]
                                                 ) > 37 else r["title"]
        print(f"{r['id']:<5} {r['date']:<12} {title:<40} {
              has_t:<12} {ingested:<22} {chunks}")


if __name__ == "__main__":
    print_status()
