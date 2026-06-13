#!/usr/bin/env python3
"""
export_omni_stats.py
====================
Aggregates statistics from the Omni Brain SQLite database into a
JSON file for the public website. No raw note content is exported —
only aggregate counts, distributions, and optionally the titles +
first-sentence excerpts of the highest-centrality output notes.

Usage:
    python export_omni_stats.py                         # default: data/brain.db → data/omni_stats.json
    python export_omni_stats.py --db path/to/brain.db --out public_html/omni_stats.json
    python export_omni_stats.py --top-notes 10          # include top-10 hub notes by PageRank
    python export_omni_stats.py --no-highlights         # skip selected notes entirely
"""

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────

def meta(row_metadata: str | None) -> dict:
    """Safely parse the JSON metadata column."""
    if not row_metadata:
        return {}
    try:
        return json.loads(row_metadata)
    except (json.JSONDecodeError, TypeError):
        return {}


def first_sentence(text: str | None, max_chars: int = 200) -> str:
    """Extract the first sentence of a note for a safe public excerpt."""
    if not text:
        return ""
    # Strip leading whitespace / markdown headers
    lines = text.strip().splitlines()
    clean = ""
    for line in lines:
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            clean = stripped
            break
    if not clean:
        return ""
    # Cut at first sentence boundary
    for sep in (". ", ".\n", "! ", "? "):
        idx = clean.find(sep)
        if 0 < idx < max_chars:
            return clean[: idx + 1]
    return clean[:max_chars].rstrip() + "…"


def month_key(date_str: str | None) -> str | None:
    """Extract YYYY-MM from a date string."""
    if not date_str or len(date_str) < 7:
        return None
    return date_str[:7]


# ── Main export ──────────────────────────────────────────────

def export(db_path: str, out_path: str, top_n: int = 8, include_highlights: bool = True):
    if not Path(db_path).exists():
        print(f"[ERROR] Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # ── 1. Corpus overview ───────────────────────────────────

    total_notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    total_words = conn.execute("SELECT COALESCE(SUM(word_count), 0) FROM notes").fetchone()[0]
    total_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    # Edge type breakdown
    edge_types = {}
    for row in conn.execute("SELECT edge_type, COUNT(*) AS cnt FROM edges GROUP BY edge_type ORDER BY cnt DESC"):
        edge_types[row["edge_type"]] = row["cnt"]

    # ── 2. Source breakdown ──────────────────────────────────

    source_counts = Counter()        # type → note count
    source_words = Counter()         # type → word count
    provenance_counts = Counter()    # output / input
    provenance_words = Counter()

    for row in conn.execute("SELECT metadata, word_count FROM notes"):
        m = meta(row["metadata"])
        src_type = m.get("type", "unknown")
        prov = m.get("provenance_role", "output")  # default to output per design
        wc = row["word_count"] or 0

        source_counts[src_type] += 1
        source_words[src_type] += wc
        provenance_counts[prov] += 1
        provenance_words[prov] += wc

    # ── 3. LLM chat statistics ───────────────────────────────

    llm_rows = conn.execute(
        "SELECT metadata, date, word_count FROM notes "
        "WHERE json_extract(metadata, '$.type') = 'llm_chat'"
    ).fetchall()

    conversations = set()
    platform_counts = Counter()      # chatgpt / claude / unknown
    role_counts = Counter()          # user / assistant
    llm_words_by_role = Counter()
    llm_monthly = Counter()          # YYYY-MM → count
    conv_titles = {}                 # conversation_id → title

    for row in llm_rows:
        m = meta(row["metadata"])
        cid = m.get("conversation_id")
        if cid:
            conversations.add(cid)
            title = m.get("conversation_title", "")
            if title and cid not in conv_titles:
                conv_titles[cid] = title

        platform = m.get("platform", "unknown")
        role = m.get("role", "unknown")
        wc = row["word_count"] or 0

        platform_counts[platform] += 1
        role_counts[role] += 1
        llm_words_by_role[role] += wc

        mk = month_key(row["date"])
        if mk:
            llm_monthly[mk] += 1

    llm_stats = {
        "total_messages": len(llm_rows),
        "total_conversations": len(conversations),
        "user_prompts": role_counts.get("user", 0),
        "assistant_responses": role_counts.get("assistant", 0) + role_counts.get("model", 0),
        "words_by_role": {
            "user": llm_words_by_role.get("user", 0),
            "assistant": llm_words_by_role.get("assistant", 0) + llm_words_by_role.get("model", 0),
        },
        "by_platform": dict(platform_counts.most_common()),
        "monthly": dict(sorted(llm_monthly.items())),
    }

    # ── 4. Temporal distribution (all notes) ─────────────────

    monthly_all = Counter()
    for row in conn.execute("SELECT date FROM notes"):
        mk = month_key(row["date"])
        if mk:
            monthly_all[mk] += 1

    # ── 5. Tag landscape ─────────────────────────────────────

    tag_counts = Counter()
    for row in conn.execute("SELECT tags FROM notes WHERE tags IS NOT NULL AND tags != ''"):
        try:
            tags = json.loads(row["tags"])
            if isinstance(tags, list):
                for t in tags:
                    if isinstance(t, str) and t.strip():
                        tag_counts[t.strip()] += 1
        except (json.JSONDecodeError, TypeError):
            pass

    # ── 6. Cluster landscape ─────────────────────────────────

    cluster_sizes = Counter()
    cluster_sample_titles = defaultdict(list)  # cluster → [titles]

    for row in conn.execute(
        "SELECT cluster, title, centrality FROM notes "
        "WHERE cluster IS NOT NULL ORDER BY centrality DESC"
    ):
        c = row["cluster"]
        cluster_sizes[c] += 1
        if len(cluster_sample_titles[c]) < 3:
            cluster_sample_titles[c].append(row["title"])

    clusters = []
    for c, size in cluster_sizes.most_common(20):
        clusters.append({
            "id": c,
            "size": size,
            "sample_titles": cluster_sample_titles[c],
        })

    # ── 7. Top hub notes (by centrality) ─────────────────────

    highlights = []
    if include_highlights and top_n > 0:
        for row in conn.execute(
            "SELECT title, centrality, tags, metadata, content, word_count "
            "FROM notes WHERE centrality > 0 "
            "ORDER BY centrality DESC LIMIT ?",
            (top_n * 3,),  # fetch extra, filter to output notes
        ):
            m = meta(row["metadata"])
            prov = m.get("provenance_role", "output")
            # Only surface the user's own writing
            if prov != "output":
                continue
            if len(highlights) >= top_n:
                break

            tags = []
            try:
                tags = json.loads(row["tags"]) if row["tags"] else []
            except (json.JSONDecodeError, TypeError):
                pass

            highlights.append({
                "title": row["title"],
                "centrality": round(row["centrality"], 6),
                "excerpt": first_sentence(row["content"]),
                "word_count": row["word_count"] or 0,
                "tags": tags[:5],
                "source_type": m.get("type", "unknown"),
            })

    # ── 8. Assemble output ───────────────────────────────────

    stats = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "corpus": {
            "total_notes": total_notes,
            "total_words": total_words,
            "total_edges": total_edges,
            "edge_types": edge_types,
        },
        "sources": {
            "by_type": {
                src_type: {"notes": count, "words": source_words[src_type]}
                for src_type, count in source_counts.most_common()
            },
            "provenance": {
                k: {"notes": provenance_counts[k], "words": provenance_words[k]}
                for k in ["output", "input"]
                if k in provenance_counts
            },
        },
        "llm": llm_stats,
        "temporal": {
            "monthly_notes": dict(sorted(monthly_all.items())),
        },
        "tags": {
            "top_30": dict(tag_counts.most_common(30)),
        },
        "clusters": clusters,
        "highlights": highlights,
    }

    # ── Write ────────────────────────────────────────────────

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    conn.close()

    print(f"[OK] Exported omni stats → {out}")
    print(f"     {total_notes:,} notes · {total_words:,} words · {total_edges:,} edges")
    print(f"     {llm_stats['total_conversations']:,} LLM conversations · {llm_stats['total_messages']:,} messages")
    print(f"     {len(clusters)} clusters · {len(highlights)} highlights")


# ── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Omni Brain statistics for the public site")
    parser.add_argument("--db", default="data/brain.db", help="Path to the Omni Brain SQLite database")
    parser.add_argument("--out", default="data/omni_stats.json", help="Output JSON path")
    parser.add_argument("--top-notes", type=int, default=8, help="Number of top hub notes to include")
    parser.add_argument("--no-highlights", action="store_true", help="Skip selected notes entirely")
    args = parser.parse_args()

    export(args.db, args.out, top_n=args.top_notes, include_highlights=not args.no_highlights)
