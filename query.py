import os
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
from logger import get_logger

load_dotenv()
log = get_logger("query")

qdrant = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_text(text: str) -> list[float]:
    log.debug(f"Embedding query text ({len(text)} chars)")
    try:
        r = openai_client.embeddings.create(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            input=text
        )
        return r.data[0].embedding
    except Exception as e:
        log.error(f"Failed to embed query: {e}")
        raise


def query_transcripts(question: str, top_k: int = 8,
                      filter_vod_id: int = None,
                      filter_speaker_global_id: str = None) -> str:
    log.info(f"Query: '{question}' | vod_filter={filter_vod_id} "
             f"speaker_filter={filter_speaker_global_id} top_k={top_k}")

    emb = embed_text(question)

    must = []
    if filter_vod_id:
        must.append(FieldCondition(
            key="vod_id", match=MatchValue(value=filter_vod_id)))
    if filter_speaker_global_id:
        must.append(FieldCondition(key="speakers",
                                   match=MatchValue(value=filter_speaker_global_id)))

    log.debug(f"Searching Qdrant with {len(must)} filter(s)...")
    try:
        results = qdrant.search(
            collection_name="transcripts",
            query_vector=emb,
            limit=top_k,
            query_filter=Filter(must=must) if must else None
        )
    except Exception as e:
        log.error(f"Qdrant search failed: {e}")
        raise

    log.info(f"Retrieved {len(results)} chunks from Qdrant")

    if not results:
        log.warning("No results returned from Qdrant")
        return "No relevant content found in transcripts."

    for i, r in enumerate(results):
        p = r.payload
        log.debug(f"  Result {i+1}: '{p['title']}' @ {p['start_ts']} "
                  f"(score={r.score:.4f}, speakers={p.get('speakers', [])})")

    context_blocks = []
    for r in results:
        p = r.payload
        context_blocks.append(
            f"Stream: {p['title']} ({p['date']})\n"
            f"Timestamp: {p['start_ts']} | {p['youtube_url']}\n"
            f"Speakers in segment: {', '.join(p.get('speakers', []))}\n"
            f"---\n{p['text']}"
        )
    context = "\n\n════════\n\n".join(context_blocks)

    system_prompt = (
        "You are an assistant that helps users find specific moments across "
        "YouTube livestream transcripts. When answering:\n"
        "- Always cite the stream title, date, and timestamp\n"
        "- Always include the full YouTube URL with the timestamp parameter\n"
        "- If the same topic appears across multiple streams, mention all of them\n"
        "- If asked about a specific speaker, focus on their lines only\n"
        "- Be concise but include short direct quotes when relevant"
    )

    log.debug("Sending context to OpenAI chat completion...")
    try:
        response = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript context:\n\n{
                    context}\n\nQuestion: {question}"}
            ],
            temperature=0.2
        )
    except Exception as e:
        log.error(f"OpenAI chat completion failed: {e}")
        raise

    answer = response.choices[0].message.content
    log.info(f"Response generated ({len(answer)} chars)")
    log.debug(f"Answer preview: {answer[:200]}...")
    return answer


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    vod_filter = None
    speaker_filter = None
    clean_args = []

    i = 0
    while i < len(args):
        if args[i] == "--vod" and i + 1 < len(args):
            vod_filter = int(args[i + 1])
            i += 2
        elif args[i] == "--speaker" and i + 1 < len(args):
            speaker_filter = args[i + 1]
            i += 2
        else:
            clean_args.append(args[i])
            i += 1

    question = " ".join(clean_args)
    if not question:
        print(
            "Usage: python query.py \"your question\" [--vod ID] [--speaker global_id]")
        sys.exit(1)

    print(query_transcripts(question,
                            filter_vod_id=vod_filter,
                            filter_speaker_global_id=speaker_filter))
