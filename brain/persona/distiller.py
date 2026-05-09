"""
Persona Distiller
-----------------
Extracts a structured intellectual DNA profile from the note corpus.

Builds a profile covering:
  1. Topical fingerprint  — weighted tag and concept distribution
  2. Stylistic markers    — sentence rhythm, vocabulary, punctuation habits
  3. Intellectual lineage — thinkers and authors cited (heuristic NER)
  4. Argument patterns    — conditional, evidential, contrastive structures
  5. Temporal arc         — how dominant topics have shifted year by year
  6. Stance map           — LLM-extracted positions on your top 10 topics
  7. Self-description     — 200-word portrait synthesised by the LLM

Profile versioning:
  Every call to build_profile() saves the new profile and archives the old
  one to data/persona_history/persona_<timestamp>.json. This lets you track
  how your intellectual identity evolves over time.

  persona.json always contains the latest profile plus:
    - previous_version_at  : ISO timestamp of the previous build
    - version              : integer counter
    - drift                : dict showing which stances changed since last build

Usage:
    distiller = PersonaDistiller(store, cfg)
    profile   = distiller.build_profile()    # builds + saves
    profile   = distiller.load_profile()     # loads latest

    history   = distiller.load_history()     # list of all past profiles
    drift     = distiller.compute_drift()    # what changed vs. last build
    distiller.print_drift_report()           # human-readable change summary
"""

import json
import logging
import math
import os
import re
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

PROFILE_PATH  = "data/persona.json"
HISTORY_DIR   = "data/persona_history"


class PersonaDistiller:
    """
    Extracts and versions a persona profile from the note corpus.

    Parameters
    ----------
    store : Store
    cfg   : dict  (llm_backend, anthropic_api_key, etc.)
    """

    def __init__(self, store, cfg: dict):
        self.store = store
        self.cfg   = cfg

    # ── Main entry ────────────────────────────────────────────────────────────

    def build_profile(self) -> dict:
        """
        Build the full persona profile and save it with versioning.
        """
        log.info("[persona] Building persona profile from corpus...")

        notes = self.store.get_all_notes()
        if not notes:
            log.warning("[persona] No notes in store.")
            return {}

        # ── THE CENTRALITY FILTER ──
        # Sort notes by centrality (importance) immediately.
        # This ensures the LLM sees your most 'connected' core ideas first.
        notes.sort(key=lambda n: getattr(n, 'centrality', 0) or 0, reverse=True)
        
        # Take the top 150 most central notes for identity synthesis to avoid timeouts.
        distilled_notes = notes[:150]

        profile = {
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "version":         1,
            "corpus_size": {
                "note_count":  len(notes),
                "total_words": sum(n.word_count() for n in notes),
            },
            # Statistical methods still use the full corpus for total accuracy
            "topical_fingerprint":  self._topical_fingerprint(notes),
            "stylistic_markers":    self._stylistic_markers(notes),
            "intellectual_lineage": self._intellectual_lineage(notes),
            "argument_patterns":    self._argument_patterns(notes),
            "temporal_arc":         self._temporal_arc(notes),
            "writing_evolution":    self._writing_evolution(notes),
            "llm_self_description": None,
            "llm_style_analysis":   None,
            "stance_map":           {},
            "previous_version_at":  None,
            "drift":                {},
        }

        # ── LLM ENRICHMENT ──
        # Use only the distilled (top 150) notes for synthesis and stance extraction.
        profile["llm_self_description"] = self._llm_synthesize_identity(distilled_notes, profile)
        profile["llm_style_analysis"]   = self._llm_deep_style(distilled_notes, profile)
        profile["stance_map"]           = self._llm_extract_stances(distilled_notes)

        # ... (rest of the archiving/saving logic remains the same)
        # Version + archive existing profile before overwriting
        existing = self.load_profile()
        if existing:
            profile["version"]           = existing.get("version", 1) + 1
            profile["previous_version_at"] = existing.get("generated_at")
            profile["drift"]             = self._diff_stances(
                existing.get("stance_map", {}),
                profile["stance_map"],
            )
            self._archive(existing)
            log.info(f"[persona] Archived v{existing.get('version', 1)} → history/")

        self.save_profile(profile)
        log.info(f"[persona] Profile v{profile['version']} built and saved.")
        return profile

    # ── Sub-extractors ────────────────────────────────────────────────────────

    def _topical_fingerprint(self, notes: list) -> dict:
        tag_counts: Counter     = Counter()
        concept_counts: Counter = Counter()
        
        # Add this blacklist to ignore administrative metadata
        BLACKLIST = {"authored", "output", "input", "generated", "synthesis", "uncategorised", "external"}
        
        stopwords = {
            "the","a","an","and","or","but","in","on","at","to","for","of",
            "with","by","from","is","was","are","were","it","this","that",
            "i","my","me","we","you","he","she","they","have","has","not",
            "be","been","will","would","could","should","also","which","when",
        }
        for note in notes:
            for tag in note.tags:
                # Add this if-statement to filter out the metadata
                if tag.lower() not in BLACKLIST:
                    tag_counts[tag] += 1
            words = re.findall(r'\b[a-z]{4,}\b', note.content.lower())
            for w in words:
                if w not in stopwords:
                    concept_counts[w] += 1

        return {
            "top_tags":      dict(tag_counts.most_common(30)),
            "top_concepts":  dict(concept_counts.most_common(50)),
            "tag_diversity": len(tag_counts),
        }

    def _stylistic_markers(self, notes: list) -> dict:
        """
        Deep stylistic analysis. Goes well beyond averages — extracts the
        full shape of the author's sentence architecture, vocabulary profile,
        rhetorical habits, and reading-level metrics.
        """
        word_lengths, sentence_lengths = [], []
        all_sentences       = []
        punctuation_counts  = Counter()
        opener_words        = Counter()   # first word of each sentence
        opener_pos          = Counter()   # grammatical class of openers
        passive_count       = 0
        hedge_count         = 0
        certainty_count     = 0
        question_count      = 0
        sentence_count      = 0
        paragraph_lengths   = []          # words per paragraph
        all_words           = []          # for bigrams / trigrams

        HEDGE_WORDS = {
            "might","may","could","perhaps","possibly","probably","seems",
            "appears","suggests","arguably","presumably","apparently",
            "tends","often","sometimes","generally","usually","typically",
            "somewhat","relatively","fairly","rather","quite","largely",
        }
        CERTAINTY_WORDS = {
            "always","never","certainly","definitely","clearly","obviously",
            "undoubtedly","inevitably","necessarily","absolutely","must",
            "will","indeed","in fact","of course","without doubt",
        }
        PASSIVE_RE = re.compile(
            r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', re.I
        )
        OPENER_PRONOUNS  = {"i","we","you","he","she","they","it","one"}
        OPENER_CONJ      = {"and","but","yet","so","nor","for","or",
                            "although","because","since","while","when",
                            "if","though","whereas","unless","until"}
        OPENER_DISCOURSE = {"this","that","these","those","such","there",
                            "here","what","which","how","why","where","who"}

        stopwords = {
            "the","a","an","and","or","but","in","on","at","to","for","of",
            "with","by","from","is","was","are","were","it","this","that",
            "i","my","me","we","you","he","she","they","have","has","not",
            "be","been","will","would","could","should","also","which","when",
            "do","did","does","had","its","their","our","what","all","more",
            "as","so","if","up","out","into","then","than","about","over",
        }

        for note in notes:
            # Paragraph lengths
            paragraphs = [p.strip() for p in note.content.split('\n\n') if p.strip()]
            for para in paragraphs:
                paragraph_lengths.append(len(para.split()))

            sentences = re.split(r'(?<=[.!?])\s+', note.content.strip())
            sentences = [s.strip() for s in sentences if len(s.split()) > 2]
            all_sentences.extend(sentences[:8])
            sentence_count += len(sentences)

            words_in_note = re.findall(r"\b[a-zA-Z']+\b", note.content)
            all_words.extend(w.lower() for w in words_in_note)
            word_lengths.extend(len(w) for w in words_in_note if w.isalpha())
            sentence_lengths.extend(len(s.split()) for s in sentences)

            # Punctuation
            for char, key in [(';', 'semicolons'), ('—', 'em_dash'),
                               ('–', 'en_dash'), (':', 'colons'),
                               ('…', 'ellipsis')]:
                n_found = note.content.count(char)
                if n_found:
                    punctuation_counts[key] += n_found
            if re.search(r'\(.*?\)', note.content):
                punctuation_counts['parentheticals'] += len(
                    re.findall(r'\(.*?\)', note.content)
                )

            # Sentence openers
            for s in sentences:
                first = re.match(r'^([A-Za-z\']+)', s)
                if first:
                    fw = first.group(1).lower()
                    opener_words[fw] += 1
                    if fw in OPENER_PRONOUNS:
                        opener_pos['pronoun_opener'] += 1
                    elif fw in OPENER_CONJ:
                        opener_pos['conjunction_opener'] += 1
                    elif fw in OPENER_DISCOURSE:
                        opener_pos['discourse_marker_opener'] += 1
                    elif fw[0].isupper() or (len(s) > 0 and s[0].isupper()):
                        opener_pos['noun_phrase_opener'] += 1
                    else:
                        opener_pos['adverb_opener'] += 1

            t_lower = note.content.lower()

            # Passive voice (heuristic)
            passive_count += len(PASSIVE_RE.findall(note.content))

            # Hedging vs. certainty
            words_lower = set(t_lower.split())
            hedge_count     += sum(1 for w in HEDGE_WORDS    if w in t_lower)
            certainty_count += sum(1 for w in CERTAINTY_WORDS if w in t_lower)

            # Questions
            question_count += note.content.count('?')

        # Core averages
        total_sentences = max(sentence_count, 1)
        total_words_n   = max(len(all_words), 1)
        avg_word   = sum(word_lengths) / max(len(word_lengths), 1)
        avg_sent   = sum(sentence_lengths) / max(len(sentence_lengths), 1)
        avg_para   = sum(paragraph_lengths) / max(len(paragraph_lengths), 1)

        # Sentence-length distribution (buckets)
        buckets = {"very_short_1_7": 0, "short_8_15": 0, "medium_16_25": 0,
                   "long_26_40": 0, "very_long_40plus": 0}
        for l in sentence_lengths:
            if l <= 7:   buckets["very_short_1_7"]   += 1
            elif l <= 15: buckets["short_8_15"]       += 1
            elif l <= 25: buckets["medium_16_25"]     += 1
            elif l <= 40: buckets["long_26_40"]       += 1
            else:         buckets["very_long_40plus"] += 1
        total_sents_bucket = max(sum(buckets.values()), 1)
        sent_distribution  = {k: round(v / total_sents_bucket, 3)
                               for k, v in buckets.items()}

        # Vocabulary richness on full corpus (not a 50k sample)
        all_tokens = [w for w in all_words if w.isalpha() and w not in stopwords]
        ttr = len(set(all_tokens)) / max(len(all_tokens), 1)

        # Hapax legomena rate (words appearing only once)
        word_freq = Counter(all_tokens)
        hapax_rate = sum(1 for v in word_freq.values() if v == 1) / max(len(word_freq), 1)

        # Flesch-Kincaid Reading Ease (approximation)
        total_syllables = sum(self._count_syllables(w) for w in all_tokens[:20000])
        fk_score = (206.835
                    - 1.015  * (total_words_n / max(total_sentences, 1))
                    - 84.6   * (total_syllables / max(total_words_n, 1)))
        fk_score = round(max(0, min(100, fk_score)), 1)

        # Grade level (Flesch-Kincaid Grade)
        fk_grade = round(
            0.39 * (total_words_n / max(total_sentences, 1))
            + 11.8 * (total_syllables / max(total_words_n, 1))
            - 15.59, 1
        )

        # Bigrams and trigrams (distinctive phrases)
        bigrams  = Counter(
            f"{all_words[i]} {all_words[i+1]}"
            for i in range(len(all_words) - 1)
            if all_words[i] not in stopwords or all_words[i+1] not in stopwords
        )
        trigrams = Counter(
            f"{all_words[i]} {all_words[i+1]} {all_words[i+2]}"
            for i in range(len(all_words) - 2)
            if all_words[i] not in stopwords
        )
        # Filter noise: only keep phrases appearing 3+ times
        sig_bigrams  = {k: v for k, v in bigrams.most_common(40)  if v >= 3}
        sig_trigrams = {k: v for k, v in trigrams.most_common(30) if v >= 3}

        # Opener analysis: top 15 non-stopword openers
        top_openers = {w: c for w, c in opener_words.most_common(30)
                       if w not in stopwords and len(w) > 2}
        top_openers = dict(list(top_openers.items())[:15])

        # Rates per 1000 words
        per_k = 1000 / max(total_words_n, 1)
        hedge_rate     = round(hedge_count     * per_k, 2)
        certainty_rate = round(certainty_count * per_k, 2)
        question_rate  = round(question_count  * per_k, 2)
        passive_rate   = round(passive_count   * per_k, 2)

        return {
            # Core metrics (preserved for backward compat)
            "avg_word_length":      round(avg_word, 2),
            "avg_sentence_length":  round(avg_sent, 2),
            "vocabulary_richness":  round(ttr, 3),
            "punctuation_style":    dict(punctuation_counts.most_common()),
            "sample_sentences":     all_sentences[:20],

            # Sentence architecture
            "sentence_length_distribution": sent_distribution,
            "avg_paragraph_length":  round(avg_para, 1),
            "sentence_opener_pos":   dict(opener_pos.most_common()),
            "top_sentence_openers":  top_openers,

            # Vocabulary depth
            "hapax_legomena_rate":   round(hapax_rate, 3),

            # Reading level
            "flesch_kincaid_ease":   fk_score,
            "flesch_kincaid_grade":  fk_grade,

            # Voice and register
            "passive_voice_per_1k":  passive_rate,
            "hedge_words_per_1k":    hedge_rate,
            "certainty_words_per_1k": certainty_rate,
            "questions_per_1k":      question_rate,
            "hedge_to_certainty_ratio": round(
                hedge_count / max(certainty_count, 1), 2
            ),

            # Linguistic fingerprint
            "signature_bigrams":  sig_bigrams,
            "signature_trigrams": sig_trigrams,
        }

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Heuristic syllable counter (no external libraries)."""
        word = word.lower().strip(".,!?;:")
        if len(word) <= 3:
            return 1
        count = len(re.findall(r'[aeiouy]+', word))
        if word.endswith('e'):
            count -= 1
        return max(1, count)

    def _intellectual_lineage(self, notes: list) -> dict:
        name_re     = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
        name_counts: Counter = Counter()
        noise = {
            "The The", "New York", "United States", "In The", "Of The",
            "Also Known", "For Example", "More Information",
        }

        # Track which tags co-occur with each cited figure (context mapping)
        figure_context: dict = {}

        for note in notes:
            found_in_note = set()
            for m in name_re.finditer(note.content):
                name = m.group(1)
                if name not in noise and len(name) > 4:
                    name_counts[name] += 1
                    found_in_note.add(name)
            for name in found_in_note:
                ctx = figure_context.setdefault(name, Counter())
                for tag in note.tags:
                    ctx[tag] += 1

        top_figures = dict(name_counts.most_common(40))
        # For top 15 figures, record their primary intellectual domain
        figure_domains = {}
        for fig in list(top_figures.keys())[:15]:
            if fig in figure_context:
                top_tag = figure_context[fig].most_common(1)
                if top_tag:
                    figure_domains[fig] = top_tag[0][0]

        return {
            "cited_figures":  top_figures,
            "figure_domains": figure_domains,  # which topic each figure connects to
        }

    def _argument_patterns(self, notes: list) -> dict:
        """
        Expanded argument-pattern extraction. Counts rhetorical moves
        across the corpus and normalises to per-note rates. Covers ~20
        distinct moves vs. the original 6.
        """
        patterns = {
            # Original 6 (preserved for compat)
            "claims_therefore":    0,
            "conditional_if_then": 0,
            "contrast_however":    0,
            "evidence_because":    0,
            "question_driven":     0,
            "list_first_second":   0,
            # New moves
            "analogy_like_as":     0,   # reasoning by analogy
            "concession_although": 0,   # acknowledging opposition
            "definition_giving":   0,   # "X is defined as / X refers to / X means"
            "example_for_instance":0,   # grounding claims with examples
            "negation_first":      0,   # "not X but Y" constructions
            "intensifier_heavy":   0,   # very/extremely/particularly dense
            "meta_cognitive":      0,   # "I think / I believe / I argue / I want to"
            "temporal_framing":    0,   # "historically / over time / in the future"
            "enumeration":         0,   # first / second / third / finally
            "rhetorical_question": 0,   # sentence ending in ? but no answer follows
            "causal_dense":        0,   # multiple causal connectors in one note
            "hedged_claim":        0,   # claim + hedge in same sentence
            "synthesis_move":      0,   # "combining / integrating / synthesising"
            "paradox_tension":     0,   # "paradox / tension / contradiction / yet"
        }

        causal_re   = re.compile(r'\b(because|since|therefore|thus|hence|so|consequently|as a result)\b', re.I)
        hedged_re   = re.compile(r'\b(might|may|perhaps|possibly|seems|appears|suggests|arguably)\b.*?\b(claim|argue|suggest|think|believe|hold|posit)\b', re.I)

        for note in notes:
            t  = note.content.lower()
            tc = note.content  # case-sensitive for some checks

            # Original patterns (updated regexes for better recall)
            if re.search(r'\b(therefore|thus,?|hence,?|it follows)\b', t):
                patterns["claims_therefore"] += 1
            if re.search(r'\bif\b.{1,80}\bthen\b', t):
                patterns["conditional_if_then"] += 1
            if re.search(r'\b(however,?|nevertheless,?|on the other hand|by contrast|conversely)\b', t):
                patterns["contrast_however"] += 1
            if re.search(r'\b(because |since |given that |in light of)\b', t):
                patterns["evidence_because"] += 1
            if note.title.endswith("?") or t.count("?") > 2:
                patterns["question_driven"] += 1
            if re.search(r'\bfirst,?\b', t) and re.search(r'\bsecond,?\b', t):
                patterns["list_first_second"] += 1

            # New patterns
            if re.search(r'\b(like |similar to |analogous to |just as |as if |as though|resembles|mirrors|parallels)\b', t):
                patterns["analogy_like_as"] += 1

            if re.search(r'\b(although|even though|while|despite|admittedly|granted|of course .{1,30}but)\b', t):
                patterns["concession_although"] += 1

            if re.search(r'\b(is defined as|refers to|can be understood as|by which I mean|what I mean is|in other words)\b', t):
                patterns["definition_giving"] += 1

            if re.search(r'\b(for example|for instance|such as|e\.g\.|namely|specifically|to illustrate|consider)\b', t):
                patterns["example_for_instance"] += 1

            if re.search(r'\bnot\b.{1,40}\bbut\b', t):
                patterns["negation_first"] += 1

            intensifier_count = len(re.findall(
                r'\b(very|extremely|particularly|especially|highly|deeply|profoundly|fundamentally|essentially|absolutely)\b', t
            ))
            if intensifier_count >= 3:
                patterns["intensifier_heavy"] += 1

            if re.search(r'\b(i think|i believe|i argue|i want to|i find|i suspect|my view|in my view|it seems to me)\b', t):
                patterns["meta_cognitive"] += 1

            if re.search(r'\b(historically|over time|in the future|over the years|long-term|eventually|gradually|used to|once was)\b', t):
                patterns["temporal_framing"] += 1

            if re.search(r'\bfirst,?\b', t) and re.search(r'\bthird,?\b', t):
                patterns["enumeration"] += 1
            elif re.search(r'\b(finally,?|lastly,?)\b', t):
                patterns["enumeration"] += 1

            questions = re.findall(r'[^.!?]*\?', tc)
            if questions:
                patterns["rhetorical_question"] += 1

            causal_hits = len(causal_re.findall(t))
            if causal_hits >= 3:
                patterns["causal_dense"] += 1

            if hedged_re.search(t):
                patterns["hedged_claim"] += 1

            if re.search(r'\b(combining|integrating|synthesising|synthesizing|bringing together|unifying|reconciling)\b', t):
                patterns["synthesis_move"] += 1

            if re.search(r'\b(paradox|tension|contradiction|yet |but also|at the same time|simultaneously|on one hand.*on the other)\b', t):
                patterns["paradox_tension"] += 1

        total = max(len(notes), 1)
        return {k: round(v / total, 3) for k, v in patterns.items()}

    def _temporal_arc(self, notes: list) -> dict:
        dated = sorted([n for n in notes if n.date], key=lambda n: n.date)
        if not dated:
            return {}
        by_year: dict = {}
        for n in dated:
            yr = str(n.date.year)
            by_year.setdefault(yr, {"count": 0, "words": 0, "tags": Counter()})
            by_year[yr]["count"] += 1
            by_year[yr]["words"] += n.word_count()
            for tag in n.tags:
                by_year[yr]["tags"][tag] += 1

        arc = {}
        for yr, data in sorted(by_year.items()):
            top = data["tags"].most_common(1)
            arc[yr] = {
                "note_count":      data["count"],
                "total_words":     data["words"],
                "dominant_topic":  top[0][0] if top else "—",
            }
        return arc

    def _writing_evolution(self, notes: list) -> dict:
        """
        Track how STYLE metrics change year over year, not just topic counts.
        Reveals whether writing has grown more complex, more hedged, more
        question-driven, etc. over time.
        """
        dated = sorted([n for n in notes if n.date], key=lambda n: n.date)
        if not dated:
            return {}

        by_year: dict = {}
        for n in dated:
            yr = str(n.date.year)
            bucket = by_year.setdefault(yr, {
                "note_count": 0, "total_words": 0,
                "sentence_lengths": [], "word_lengths": [],
                "hedge_count": 0, "question_count": 0,
                "certainty_count": 0, "tags": Counter(),
            })
            bucket["note_count"] += 1
            bucket["total_words"] += n.word_count()

            sentences = re.split(r'(?<=[.!?])\s+', n.content.strip())
            sentences = [s for s in sentences if len(s.split()) > 2]
            bucket["sentence_lengths"].extend(len(s.split()) for s in sentences)

            words = re.findall(r'\b[a-zA-Z]+\b', n.content)
            bucket["word_lengths"].extend(len(w) for w in words if w.isalpha())

            t = n.content.lower()
            bucket["hedge_count"]     += len(re.findall(
                r'\b(might|may|perhaps|possibly|seems|appears|suggests|arguably|presumably)\b', t))
            bucket["question_count"]  += t.count('?')
            bucket["certainty_count"] += len(re.findall(
                r'\b(always|never|certainly|definitely|clearly|obviously|must|absolutely)\b', t))
            for tag in n.tags:
                bucket["tags"][tag] += 1

        arc = {}
        for yr, data in sorted(by_year.items()):
            sl = data["sentence_lengths"]
            wl = data["word_lengths"]
            wc = max(data["total_words"], 1)
            top_tag = data["tags"].most_common(1)
            arc[yr] = {
                "note_count":           data["note_count"],
                "total_words":          data["total_words"],
                "avg_sentence_length":  round(sum(sl) / max(len(sl), 1), 1),
                "avg_word_length":      round(sum(wl) / max(len(wl), 1), 2),
                "hedges_per_1k_words":  round(data["hedge_count"] * 1000 / wc, 2),
                "questions_per_1k":     round(data["question_count"] * 1000 / wc, 2),
                "certainty_per_1k":     round(data["certainty_count"] * 1000 / wc, 2),
                "dominant_topic":       top_tag[0][0] if top_tag else "—",
            }
        return arc

    def _llm_deep_style(self, notes: list, profile: dict) -> dict:
        """
        Feed actual writing samples to the LLM for qualitative fingerprinting.
        Returns structured JSON with deep stylistic observations that statistics
        alone cannot capture: rhetorical personality, intellectual moves,
        what the writing reveals about the author's character.
        """
        # Select a diverse sample: top 10 by centrality + 10 random spread
        import random
        random.seed(42)
        top_notes    = notes[:10]
        other_notes  = notes[10:]
        sample_pool  = top_notes + random.sample(other_notes, min(10, len(other_notes)))

        # Build a rich sample: title + first 300 chars of each note
        samples = "\n\n---\n".join(
            f"[{n.title}]\n{n.content[:350].strip()}"
            for n in sample_pool
        )

        sm = profile.get("stylistic_markers", {})
        ap = profile.get("argument_patterns", {})

        # Sort argument patterns to find top 5 most frequent
        top_moves = sorted(ap.items(), key=lambda x: x[1], reverse=True)[:5]
        top_moves_str = ", ".join(f"{k.replace('_', ' ')} ({v:.1%})" for k, v in top_moves)

        prompt = f"""You are a literary analyst and forensic stylometrist.
Analyse the writing samples below from a personal knowledge corpus and produce a
DEEP stylistic profile. Think: if you had to testify in court that these texts
were written by the same person as another document, what specific features would
you cite?

CORPUS STATISTICS:
- {profile['corpus_size']['note_count']} notes, {profile['corpus_size']['total_words']:,} total words
- Avg sentence length: {sm.get('avg_sentence_length', '?')} words
- Flesch-Kincaid reading ease: {sm.get('flesch_kincaid_ease', '?')} / 100
- Hedge-to-certainty ratio: {sm.get('hedge_to_certainty_ratio', '?')} (>1 = more hedging)
- Top rhetorical moves: {top_moves_str}

WRITING SAMPLES:
{samples}

Respond ONLY in valid JSON (no markdown fences, no explanation outside JSON):
{{
  "voice_character": "2-3 sentences describing the fundamental voice and personality on the page",
  "intellectual_moves": ["list of 5-7 specific, named argumentative or rhetorical moves this author habitually makes, e.g. 'opens with a rhetorical question then defers its answer' or 'uses technical jargon then immediately defines it'"],
  "sentence_personality": "specific description of HOW sentences are built — not length stats, but structural tendencies (e.g. front-loaded subordinate clauses, mid-sentence pivots, list-as-argument)",
  "vocabulary_character": "what the word choices reveal — register (formal/conversational), domain biases, favourite modifiers, words used idiosyncratically",
  "what_the_writing_conceals": "what the author systematically avoids, hedges around, or never directly states — gaps and silences",
  "distinctive_tics": ["3-5 very specific verbal habits — exact phrases, punctuation patterns, or structural choices that would appear in a forensic stylometric match"],
  "intellectual_posture": "how the author positions themselves relative to their sources and reader — authority, doubt, collaboration, challenge",
  "emotional_register": "the emotional temperature and what emotions leak through despite intellectual framing",
  "writing_under_pressure": "what the writing looks like in its least polished moments — when the author drops the rhetorical control",
  "one_sentence_fingerprint": "a single forensic sentence that would uniquely identify this author's writing style to a literary expert"
}}"""

        try:
            raw = self._llm_call(prompt, max_tokens=1200)
            # Strip any accidental markdown fences
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r'^```[a-z]*\n?', '', raw)
                raw = re.sub(r'\n?```$', '', raw)
            return json.loads(raw)
        except Exception as e:
            log.warning(f"[persona] LLM deep style analysis failed: {e}")
            return {"error": str(e)}

    # ── LLM enrichment ───────────────────────────────────────────────────────

    def _llm_synthesize_identity(self, notes: list, profile: dict) -> str:
        top_tags     = list(profile["topical_fingerprint"]["top_tags"].keys())[:15]
        top_concepts = list(profile["topical_fingerprint"]["top_concepts"].keys())[:20]
        cited        = list(profile["intellectual_lineage"]["cited_figures"].keys())[:15]
        sm           = profile.get("stylistic_markers", {})

        # Include a real writing sample (3 notes, full opening paragraph each)
        sample_texts = "\n\n---\n".join(
            f"[{n.title}]\n{n.content[:400].strip()}"
            for n in notes[:3]
        )

        # Surface the most distinctive argument moves
        ap       = profile.get("argument_patterns", {})
        top_moves = sorted(ap.items(), key=lambda x: x[1], reverse=True)[:4]
        moves_str = ", ".join(f"{k.replace('_',' ')} ({v:.0%})" for k, v in top_moves)

        prompt = f"""You are analysing a personal knowledge corpus to describe the
intellectual identity of the person who wrote it.

Corpus statistics:
- {profile['corpus_size']['note_count']} notes, {profile['corpus_size']['total_words']:,} words
- Reading level: Flesch-Kincaid ease {sm.get('flesch_kincaid_ease', '?')}/100, grade {sm.get('flesch_kincaid_grade', '?')}
- Avg sentence: {sm.get('avg_sentence_length', '?')} words; hedge/certainty ratio: {sm.get('hedge_to_certainty_ratio', '?')}
- Most common tags: {', '.join(top_tags)}
- Most frequent concepts: {', '.join(top_concepts[:15])}
- Figures mentioned most: {', '.join(cited) if cited else 'none detected'}
- Top argument moves: {moves_str}
- Sample note titles: {'; '.join(n.title for n in notes[:20])}

ACTUAL WRITING SAMPLES:
{sample_texts}

Write a 200-word intellectual profile in second person ("You are someone who...").
Be specific and concrete — name their actual intellectual obsessions, the traditions
they're in dialogue with, what questions genuinely drive them, and how their writing
style reflects their mode of thinking. Avoid generic phrases."""

        try:
            return self._llm_call(prompt, max_tokens=500)
        except Exception as e:
            log.warning(f"[persona] LLM identity synthesis failed: {e}")
            return (f"Corpus of {profile['corpus_size']['note_count']} notes "
                    f"on: {', '.join(top_tags[:5])}")

    def _llm_extract_stances(self, notes: list) -> dict:
        all_tags: Counter = Counter()
        # Add the same blacklist here[cite: 5]
        BLACKLIST = {"authored", "output", "input", "generated", "synthesis", "uncategorised", "external"}

        for n in notes:
            for t in n.tags:
                # Filter the tags before counting[cite: 5]
                if t.lower() not in BLACKLIST:
                    all_tags[t] += 1

        top_topics = [t for t, _ in all_tags.most_common(10)]
        stances    = {}
        for topic in top_topics:
            topic_notes = [n for n in notes if topic in n.tags][:5]
            if not topic_notes:
                continue
            context = "\n\n---\n".join(
                f"[{n.title}]\n{n.short_content(400)}" for n in topic_notes
            )
            prompt = f"""Based on these notes about "{topic}", describe the author's
intellectual stance in 1-2 sentences. Be specific about their actual position.

Notes:
{context}

Just the stance description, no preamble."""
            try:
                stances[topic] = self._llm_call(prompt, max_tokens=150).strip()
            except Exception as e:
                log.debug(f"[persona] Stance failed for '{topic}': {e}")

        return stances

    # ── Versioning ────────────────────────────────────────────────────────────

    def _archive(self, old_profile: dict):
        """Save a copy of old_profile to data/persona_history/."""
        Path(HISTORY_DIR).mkdir(parents=True, exist_ok=True)
        ts   = (old_profile.get("generated_at", datetime.now(timezone.utc).isoformat())
                .replace(":", "-").replace("+", "Z")[:19])
        ver  = old_profile.get("version", 0)
        path = Path(HISTORY_DIR) / f"persona_v{ver}_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(old_profile, f, indent=2, default=str)

    def _diff_stances(self, old: dict, new: dict) -> dict:
        """
        Compute which stances changed between two profiles.
        Returns {"topic": {"old": str, "new": str}} for changed topics.
        """
        drift = {}
        all_topics = set(old) | set(new)
        for topic in all_topics:
            o = old.get(topic, "")
            n = new.get(topic, "")
            if o != n:
                drift[topic] = {"old": o, "new": n}
        return drift

    def load_history(self) -> list:
        """Return all archived profiles, sorted oldest first."""
        hist_dir = Path(HISTORY_DIR)
        if not hist_dir.exists():
            return []
        profiles = []
        for p in sorted(hist_dir.glob("persona_v*.json")):
            try:
                with open(p) as f:
                    profiles.append(json.load(f))
            except Exception:
                pass
        return profiles

    def compute_drift(self) -> dict:
        """
        Compare the current profile to the previous one.
        Returns a drift dict (or empty dict if only one version exists).
        """
        current = self.load_profile()
        if not current:
            return {}
        history = self.load_history()
        if not history:
            return {}
        previous = history[-1]
        return self._diff_stances(
            previous.get("stance_map", {}),
            current.get("stance_map",  {}),
        )

    def print_drift_report(self):
        """Print a human-readable summary of how your thinking has changed."""
        drift   = self.compute_drift()
        current = self.load_profile()
        if not current:
            print("No persona profile found. Run: python main.py persona build")
            return

        v    = current.get("version", 1)
        prev = current.get("previous_version_at", "—")
        now  = current.get("generated_at", "—")

        print(f"\n{'='*60}")
        print(f"  PERSONA DRIFT REPORT")
        print(f"  v{v-1} ({prev[:10]}) → v{v} ({now[:10]})")
        print(f"{'='*60}\n")

        if not drift:
            print("  No stance changes detected since last build.")
            return

        print(f"  {len(drift)} topic(s) show changed stances:\n")
        for topic, change in drift.items():
            print(f"  [{topic}]")
            if change["old"]:
                print(f"    Before: {change['old']}")
            else:
                print(f"    Before: (new topic)")
            if change["new"]:
                print(f"    After:  {change['new']}")
            else:
                print(f"    After:  (topic dropped)")
            print()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_profile(self, profile: dict, path: str = PROFILE_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, default=str)
        log.info(f"[persona] Profile saved to {path}")

    def load_profile(self, path: str = PROFILE_PATH) -> Optional[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    # ── LLM backend ──────────────────────────────────────────────────────────

    def _llm_call(self, prompt: str, max_tokens: int = 512) -> str:
        backend = self.cfg.get("llm_backend", "claude")
        if backend == "ollama":
            base  = self.cfg.get("ollama_base_url", "http://localhost:11434").rstrip("/")
            model = self.cfg.get("ollama_model", "mistral")
            payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
            req = urllib.request.Request(
                f"{base}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())["response"]
        else:
            api_key = self.cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
            model   = self.cfg.get("claude_model", "claude-haiku-4-5-20251001")
            payload = json.dumps({
                "model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=payload,
                headers={"Content-Type": "application/json",
                         "x-api-key": api_key,
                         "anthropic-version": "2023-06-01"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())["content"][0]["text"]
