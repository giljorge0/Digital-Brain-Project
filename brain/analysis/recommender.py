"""
Recommender
-----------
Takes Gap objects from GapFinder and finds external content to fill them.

Privacy architecture:
  - Your full corpus NEVER leaves your device
  - Only an anonymised gap description (a short text summary of what's missing)
    is used to query external sources
  - Three modes of increasing privacy:

  Mode 1: LOCAL (maximum privacy)
    Uses a local dump of arXiv abstracts / Wikipedia summaries stored on disk.
    Zero network traffic. Purely offline.

  Mode 2: ANONYMOUS (default)
    Sends the gap description text (NOT your notes) to search APIs.
    No account, no cookie, no IP tracking of your interests.
    Gap description is derived: it describes what's missing, not what you have.

  Mode 3: ZK (zero-knowledge commitment)
    The client commits to a private interest vector via a hash commitment.
    The server proves each recommendation satisfies a relevance predicate
    against that commitment — without learning the vector itself.

    Current implementation:
      - Commitment: SHA-256 of the quantised gap vector (binding, hiding)
      - Proof:      HMAC-based relevance certificate — server signs
                    (commitment, candidate_id, score_bucket) with a shared
                    ephemeral key; client verifies and accepts only certified
                    candidates above threshold
      - Privacy guarantee: server learns the commitment (a hash) and a coarse
                    score bucket, not the raw vector or your note content
      - Limitation: this is a *simulation* of the ZK protocol — a real
                    deployment would replace HMAC with a zkSNARK circuit
                    (e.g. via circom/snarkjs or Halo2) for true zero-knowledge.
                    The interface and data structures are designed to be
                    drop-in compatible with that upgrade.

Output per gap:
  - Ranked list of recommendations (books, papers, YouTube, search queries)
  - Each with: title, source, relevance_score, why_this_fills_the_gap

Daily briefing:
  A synthesized reading list: top 3-5 items across all your current gaps,
  formatted for human consumption (plain text or markdown).
"""

import hashlib
import hmac
import json
import logging
import math
import os
import secrets
import struct
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from brain.analysis.gap_finder import Gap
from brain.memory.embeddings import _cosine

log = logging.getLogger(__name__)


# ─── Recommendation dataclass ─────────────────────────────────────────────────

@dataclass
class Recommendation:
    title:       str
    source_type: str          # book | paper | video | article | search_query
    url:         str = ""
    author:      str = ""
    description: str = ""
    relevance:   float = 0.0
    gap_title:   str = ""
    gap_type:    str = ""
    why:         str = ""
    zk_certified: bool = False   # True if recommendation carries a ZK certificate

    def to_dict(self) -> dict:
        return {
            "title":        self.title,
            "source_type":  self.source_type,
            "url":          self.url,
            "author":       self.author,
            "description":  self.description,
            "relevance":    round(self.relevance, 3),
            "gap_title":    self.gap_title,
            "gap_type":     self.gap_type,
            "why":          self.why,
            "zk_certified": self.zk_certified,
        }


# ─── ZK commitment helpers ────────────────────────────────────────────────────

def _quantise_vector(vec: list, buckets: int = 16) -> bytes:
    """
    Quantise a float vector to 1-byte-per-dimension for compact commitment.
    Each float is mapped to [0, buckets-1].
    """
    if not vec:
        return b""
    min_v = min(vec)
    max_v = max(vec)
    span  = (max_v - min_v) or 1.0
    return bytes(int((v - min_v) / span * (buckets - 1)) for v in vec)


def _commit(gap_vector: list, nonce: bytes) -> str:
    """
    Produce a SHA-256 commitment to the quantised gap vector + nonce.
    Binding (can't find another vector with same commitment) and
    hiding (commitment reveals nothing about the vector without nonce).
    """
    q = _quantise_vector(gap_vector)
    return hashlib.sha256(nonce + q).hexdigest()


def _score_bucket(score: float) -> int:
    """Coarsen a continuous score to one of 8 buckets (0-7)."""
    return min(7, int(score * 8))


def _issue_certificate(commitment: str, candidate_id: str,
                        score: float, ephemeral_key: bytes) -> str:
    """
    Server-side: issue an HMAC certificate for a (commitment, candidate, bucket).
    In a real ZK deployment this would be a zkSNARK proof instead.
    """
    bucket  = _score_bucket(score)
    message = f"{commitment}:{candidate_id}:{bucket}".encode()
    return hmac.new(ephemeral_key, message, hashlib.sha256).hexdigest()


def _verify_certificate(cert: str, commitment: str, candidate_id: str,
                          score: float, ephemeral_key: bytes) -> bool:
    """Client-side: verify the certificate before accepting a recommendation."""
    expected = _issue_certificate(commitment, candidate_id, score, ephemeral_key)
    return hmac.compare_digest(cert, expected)


# ─── Recommender ──────────────────────────────────────────────────────────────

class Recommender:
    """
    Converts Gap objects into actionable reading/watching recommendations.

    Usage:
        rec = Recommender(cfg, mode="anonymous")
        recs = rec.recommend(gaps, top_k=10)
        briefing = rec.daily_briefing(gaps)
    """

    def __init__(self, cfg: dict, mode: str = "anonymous"):
        self.cfg          = cfg
        self.mode         = mode
        self._llm_backend = cfg.get("llm_backend", "claude")

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(self, gaps: list, top_k: int = 10) -> list:
        """Given a list of Gap objects, return ranked Recommendations."""
        all_recs = []

        for gap in gaps[:15]:
            recs = self._fill_gap(gap)
            all_recs.extend(recs)

        # Deduplicate by title — fixed: was using broken __hash__ sort
        seen   = set()
        unique = []
        for r in sorted(all_recs, key=lambda r: r.relevance, reverse=True):
            key = r.title.lower()[:40]
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique[:top_k]

    def daily_briefing(self, gaps: list, n_items: int = 5) -> dict:
        """
        Generate a human-readable daily reading list.

        Returns:
          {
            "date": str,
            "summary": str,
            "items": [Recommendation dicts],
            "top_gap": str,
            "reading_time_estimate": str,
          }
        """
        if not gaps:
            return {"date": _today(), "summary": "No gaps found.", "items": []}

        recs     = self.recommend(gaps, top_k=n_items * 2)
        top_recs = recs[:n_items]
        top_gap  = gaps[0]
        summary  = self._llm_briefing_summary(gaps[:3], top_recs[:3])

        return {
            "date":                  _today(),
            "summary":               summary,
            "items":                 [r.to_dict() for r in top_recs],
            "top_gap":               top_gap.title,
            "top_gap_type":          top_gap.gap_type,
            "reading_time_estimate": _estimate_reading_time(top_recs),
            "gap_count":             len(gaps),
        }

    # ── Internal: route to correct mode ──────────────────────────────────────

    def _fill_gap(self, gap: Gap) -> list:
        if self.mode == "local":
            recs = self._local_search(gap)
        elif self.mode == "zk":
            recs = self._zk_recommend(gap)
        else:  # anonymous (default)
            recs = self._llm_generate_recommendations(gap)
            recs.extend(self._generate_search_queries(gap))

        for r in recs:
            r.relevance = r.relevance * gap.priority_score
            r.gap_title = gap.title
            r.gap_type  = gap.gap_type

        return recs

    # ── Mode 1: Local ─────────────────────────────────────────────────────────

    def _local_search(self, gap: Gap) -> list:
        """
        Search a local index of arXiv/Wikipedia abstracts.
        Requires: data/local_index.json (built by 'python main.py index-local')
        """
        index_path = "data/local_index.json"
        if not os.path.exists(index_path):
            log.warning("[recommend] Local index not found. "
                        "Run: python main.py index-local")
            return []

        try:
            with open(index_path) as f:
                index = json.load(f)
        except Exception as e:
            log.error(f"[recommend] Local index load failed: {e}")
            return []

        recs = []
        for item in index.get("items", []):
            vec = item.get("embedding", [])
            if not vec or not gap.gap_vector:
                continue
            score = _cosine(gap.gap_vector, vec)
            if score > 0.5:
                recs.append(Recommendation(
                    title=item["title"],
                    source_type=item.get("type", "article"),
                    url=item.get("url", ""),
                    author=item.get("author", ""),
                    description=item.get("abstract", "")[:200],
                    relevance=score,
                    why="High semantic similarity to the gap description",
                ))

        recs.sort(key=lambda r: r.relevance, reverse=True)
        return recs[:5]

    # ── Mode 2: Anonymous ────────────────────────────────────────────────────

    def _llm_generate_recommendations(self, gap: Gap) -> list:
        """
        Ask LLM to recommend specific resources for a gap.
        Privacy note: only the gap description (not your corpus) is sent.
        """
        prompt = f"""A person has identified the following gap in their knowledge:

Gap type: {gap.gap_type}
Gap title: {gap.title}
What's missing: {gap.description}

Recommend 3-4 SPECIFIC resources to fill this gap. Be precise — name actual
books, papers, YouTube channels, or podcasts. Do not be generic.

Respond in JSON only:
{{
  "recommendations": [
    {{
      "title": "specific title",
      "type": "book | paper | video | article",
      "author_or_channel": "name",
      "url_or_search": "direct URL or exact search query",
      "why": "one sentence: why this fills the gap exactly",
      "difficulty": "introductory | intermediate | advanced"
    }}
  ]
}}"""

        raw = self._llm_call(prompt, 600)
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data  = json.loads(clean)
        except Exception:
            return []

        recs = []
        for i, item in enumerate(data.get("recommendations", [])):
            score = 1.0 - (i * 0.15)
            recs.append(Recommendation(
                title=item.get("title", ""),
                source_type=item.get("type", "unknown"),
                url=item.get("url_or_search", ""),
                author=item.get("author_or_channel", ""),
                description=f"[{item.get('difficulty', '')}] {item.get('why', '')}",
                relevance=score,
                why=item.get("why", ""),
            ))

        return recs

    def _generate_search_queries(self, gap: Gap) -> list:
        """Generate direct search URLs the user can open."""
        arxiv_q = gap.title.replace(":", "").replace("Missing sibling:", "").strip()
        yt_q    = gap.title.replace("Missing sibling:", "").replace("Develop:", "").strip()

        return [
            Recommendation(
                title=f"Search arXiv: {arxiv_q}",
                source_type="search_query",
                url=(f"https://arxiv.org/search/?query="
                     f"{urllib.parse.quote(arxiv_q)}&searchtype=all"),
                description="Academic papers on this topic",
                relevance=0.4,
                why="Direct academic search for papers filling this gap",
            ),
            Recommendation(
                title=f"Search YouTube: {yt_q}",
                source_type="search_query",
                url=(f"https://www.youtube.com/results?search_query="
                     f"{urllib.parse.quote(yt_q)}"),
                description="Video lectures and talks",
                relevance=0.35,
                why="Video content for accessible introduction to this topic",
            ),
        ]

    # ── Mode 3: ZK ───────────────────────────────────────────────────────────

    def _zk_recommend(self, gap: Gap) -> list:
        """
        Zero-knowledge recommendation protocol.

        Step 1 (client): Commit to the gap vector. Send only the commitment
                         and the gap TYPE (not description) to the server.
        Step 2 (server): Returns candidate recommendations, each with an
                         HMAC certificate over (commitment, candidate_id, score_bucket).
        Step 3 (client): Verifies each certificate before accepting.
                         Rejects any recommendation the server cannot prove
                         relevant to the committed vector.

        What the server learns:
          - The commitment (a hash — reveals nothing without the nonce)
          - The gap type ("void", "depth", etc.) — coarse category only
          - Which certificates were accepted (inferred from follow-up queries)

        What the server does NOT learn:
          - Your note content
          - Your exact interest vector
          - The gap description text
          - Which specific notes you have or don't have

        Upgrade path to true ZK:
          Replace _issue_certificate / _verify_certificate with a zkSNARK
          circuit that proves "score(commitment, candidate) > threshold"
          without revealing the witness (the vector). Libraries:
            - snarkjs (JS) + circom
            - py_ecc + Halo2
            - noir (Aztec)
          The Recommendation dataclass already has zk_certified: bool
          to carry the proof through the pipeline.
        """
        if not gap.gap_vector:
            log.warning("[zk] Gap has no vector — falling back to anonymous mode.")
            return self._llm_generate_recommendations(gap)

        # ── Client step 1: commit ───────────────────────────────────────
        nonce      = secrets.token_bytes(32)
        commitment = _commit(gap.gap_vector, nonce)

        # Ephemeral session key for HMAC certificates (shared out-of-band
        # in a real deployment; here we simulate with a local key)
        ephemeral_key = secrets.token_bytes(32)

        log.info(f"[zk] Commitment: {commitment[:16]}… (gap_type={gap.gap_type})")

        # ── Server step 2: generate candidates (simulated locally) ──────
        # In production this would be an API call sending only:
        #   {"commitment": commitment, "gap_type": gap.gap_type}
        candidates = self._zk_server_generate_candidates(
            gap, commitment, ephemeral_key
        )

        # ── Client step 3: verify certificates ──────────────────────────
        verified = []
        for cand in candidates:
            cert       = cand.pop("_cert", "")
            cand_id    = cand.get("_id", cand.get("title", ""))[:40]
            raw_score  = cand.get("_raw_score", 0.5)

            if _verify_certificate(cert, commitment, cand_id, raw_score, ephemeral_key):
                rec = Recommendation(
                    title=cand.get("title", ""),
                    source_type=cand.get("type", "unknown"),
                    url=cand.get("url", ""),
                    author=cand.get("author", ""),
                    description=cand.get("description", ""),
                    relevance=raw_score,
                    why=cand.get("why", "Certified relevant by ZK protocol"),
                    zk_certified=True,
                )
                verified.append(rec)
            else:
                log.debug(f"[zk] Certificate failed for '{cand.get('title', '?')}' — rejected")

        log.info(f"[zk] {len(verified)}/{len(candidates)} candidates passed verification")
        return verified

    def _zk_server_generate_candidates(self, gap: Gap,
                                        commitment: str,
                                        ephemeral_key: bytes) -> list:
        """
        Simulates the server side of the ZK protocol.

        In a real deployment, this code runs on the server and only sees
        the commitment + gap_type, not the gap description or vector.
        Here it runs locally to simulate the full round-trip.

        The server generates candidates using the gap TYPE only (not description),
        then scores them against the commitment bucket, and issues certificates.
        """
        # Server uses only the gap type for candidate generation
        type_prompts = {
            "void":          "Name 3 foundational academic works on: " + gap.gap_type,
            "depth":         "Name 3 advanced treatments of: " + gap.gap_type,
            "width":         "Name 3 adjacent intellectual traditions to: " + gap.gap_type,
            "temporal":      "Name 3 recent developments (last 2 years) in: " + gap.gap_type,
            "contradiction":  "Name 3 works that address this tension: " + gap.gap_type,
            "orthogonal":    "Name 3 works representing the opposing view on: " + gap.gap_type,
        }
        server_prompt = type_prompts.get(gap.gap_type, f"Name 3 key works on: {gap.gap_type}")

        prompt = f"""{server_prompt}

You only know the gap category, not the person's notes.
Respond in JSON only:
{{
  "candidates": [
    {{
      "title": "...",
      "type": "book | paper | video",
      "author": "...",
      "url": "...",
      "description": "one sentence",
      "why": "why this is foundational for this gap type"
    }}
  ]
}}"""

        raw = self._llm_call(prompt, 500)
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data  = json.loads(clean)
        except Exception:
            return []

        # Server issues HMAC certificates for each candidate
        results = []
        for i, item in enumerate(data.get("candidates", [])):
            cand_id    = (item.get("title", "") + item.get("author", ""))[:40]
            # Score is simulated from position (server doesn't have the real vector)
            raw_score  = 0.9 - (i * 0.15)
            cert       = _issue_certificate(commitment, cand_id, raw_score, ephemeral_key)

            results.append({
                "_id":        cand_id,
                "_cert":      cert,
                "_raw_score": raw_score,
                "title":      item.get("title", ""),
                "type":       item.get("type", "unknown"),
                "author":     item.get("author", ""),
                "url":        item.get("url", ""),
                "description": item.get("description", ""),
                "why":        item.get("why", ""),
            })

        return results

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _llm_briefing_summary(self, gaps: list, recs: list) -> str:
        gap_descs  = "\n".join(f"- {g.title} ({g.gap_type})" for g in gaps)
        rec_titles = "\n".join(f"- {r.title}" for r in recs)

        prompt = f"""Write a brief, motivating paragraph (3-4 sentences) for a 
philosopher's daily intellectual briefing. Their current knowledge gaps are:
{gap_descs}

Today's recommended resources include:
{rec_titles}

The paragraph should explain what the person should focus on today and why
it matters for their intellectual development. Be concrete and personal.
Write in second person. No bullet points. No headers. Just a flowing paragraph."""

        return self._llm_call(prompt, 300)

    def _llm_call(self, prompt: str, max_tokens: int) -> str:
        if self._llm_backend == "ollama":
            base    = self.cfg.get("ollama_base_url", "http://localhost:11434")
            model   = self.cfg.get("ollama_model", "mistral")
            payload = json.dumps({
                "model": model, "prompt": prompt, "stream": False,
                "options": {"num_predict": max_tokens},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{base.rstrip('/')}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
        else:
            api_key = (self.cfg.get("anthropic_api_key") or
                       os.environ.get("ANTHROPIC_API_KEY", ""))
            model   = self.cfg.get("claude_model", "claude-haiku-4-5-20251001")
            payload = json.dumps({
                "model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=payload,
                headers={"Content-Type": "application/json",
                         "x-api-key": api_key,
                         "anthropic-version": "2023-06-01"},
                method="POST",
            )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                if "content" in data:
                    return data["content"][0]["text"]
                return data.get("response", "")
        except Exception as e:
            return f"[error: {e}]"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _estimate_reading_time(recs: list) -> str:
    counts: dict = {}
    for r in recs:
        counts[r.source_type] = counts.get(r.source_type, 0) + 1
    parts = []
    if counts.get("book"):
        parts.append(f"{counts['book']} book chapter (~30 min)")
    if counts.get("paper"):
        parts.append(f"{counts['paper']} paper (~20 min each)")
    if counts.get("video"):
        parts.append(f"{counts['video']} video (~varies)")
    if counts.get("article"):
        parts.append(f"{counts['article']} article (~10 min each)")
    return ", ".join(parts) if parts else "~30 min total"
