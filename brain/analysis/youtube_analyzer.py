"""
YouTube History Deep Analyzer
------------------------------
Performs temporal, behavioral, and cognitive analysis on raw Google Takeout
YouTube data to reconstruct an intellectual portrait of who you were,
what the algorithm was pushing, and how those two things interacted.

Analysis layers:
  1. TEMPORAL PERSONA TIMELINE
     Monthly snapshots: dominant channels/topics, watch velocity (videos/day),
     topic diversity, time-of-day profile. Shows who you were via YouTube at
     each point in time.

  2. TOPIC DRIFT DETECTION
     Identifies inflection points where your viewing patterns shifted:
     - when you started or stopped a topic cluster
     - how fast the shift happened (gradual drift vs sudden pivot)
     - what the triggering month looked like

  3. ALGORITHM → SEARCH SYNERGY
     The key insight: topics appear in your watch history (algorithm pushing)
     weeks before they appear in your search history (active seeking).
     Detects these lag patterns and estimates your typical passive→active
     conversion time per topic cluster.

  4. TIME ALLOCATION ANALYSIS
     How you distribute watching across:
     - time of day (morning / afternoon / evening / night)
     - weekday vs weekend
     - binge sessions (many videos clustered in time)
     - velocity over years (were you watching more in 2022 than 2024?)

  5. PLAYLIST CRYSTALLIZATION
     If playlist data available: gap between first watch of a topic and
     first save to playlist = interest crystallization time.
     Short gap = immediate resonance. Long gap = slow burn.

Input (Google Takeout):
  watch-history.json     Takeout/YouTube/history/watch-history.json
  search-history.json    Takeout/YouTube/history/search-history.json
  playlists/*.json       Takeout/YouTube/playlists/

Output:
  YouTubePersonaReport — printable, JSON-exportable, integrates with persona.json

Usage:
  from brain.analysis.youtube_analyzer import YouTubeAnalyzer
  analyzer = YouTubeAnalyzer(cfg)
  report   = analyzer.analyze(
      watch_path  = Path("Takeout/YouTube/history/watch-history.json"),
      search_path = Path("Takeout/YouTube/history/search-history.json"),  # optional
      playlist_dir= Path("Takeout/YouTube/playlists/"),                   # optional
  )
  report.print_summary()
  report.save("data/youtube_report.json")
  report.integrate_with_persona("data/persona.json")
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("brain.youtube")


# ─── Raw event types ──────────────────────────────────────────────────────────

@dataclass
class VideoEvent:
    title:     str
    channel:   str
    url:       str
    video_id:  str
    timestamp: datetime
    topics:    list = field(default_factory=list)   # filled after categorization

    @property
    def month(self) -> str:
        return self.timestamp.strftime("%Y-%m")

    @property
    def week(self) -> str:
        return self.timestamp.strftime("%Y-W%V")

    @property
    def hour(self) -> int:
        return self.timestamp.hour

    @property
    def weekday(self) -> int:
        return self.timestamp.weekday()   # 0=Mon … 6=Sun


@dataclass
class SearchEvent:
    query:     str
    timestamp: datetime
    topics:    list = field(default_factory=list)

    @property
    def month(self) -> str:
        return self.timestamp.strftime("%Y-%m")


@dataclass
class PlaylistEvent:
    playlist_name: str
    video_id:      str
    title:         str
    added_at:      Optional[datetime]


# ─── Analysis result types ────────────────────────────────────────────────────

@dataclass
class TopicWindow:
    """Aggregated stats for a calendar month."""
    month:            str
    video_count:      int
    top_channels:     list         # [(channel, count), ...]
    top_topics:       list         # [(topic, count), ...]
    watch_velocity:   float        # videos per day (avg in this month)
    diversity_score:  float        # 0-1: how spread across channels
    tod_profile:      dict         # {morning:%, afternoon:%, evening:%, night:%}
    weekend_fraction: float        # 0-1: fraction watched on weekends


@dataclass
class DriftEvent:
    """A detected inflection point in viewing patterns."""
    month:         str
    description:   str
    topics_gained: list
    topics_lost:   list
    magnitude:     float           # 0-1: how dramatic the shift was


@dataclass
class SynergySignal:
    """Algorithm-to-search synergy: topic watched passively then searched actively."""
    topic:           str
    first_watched:   str           # month
    first_searched:  str           # month
    lag_weeks:       float
    watch_count:     int
    search_count:    int


@dataclass
class BingeSession:
    start:      datetime
    end:        datetime
    videos:     list               # list of video titles
    topics:     list               # dominant topics

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60

    @property
    def video_count(self) -> int:
        return len(self.videos)


# ─── Canonical topic map (channel → topic cluster) ───────────────────────────
# Covers the most common serious/intellectual YouTube channels.
# Falls back to keyword heuristics for unrecognized channels.

_CHANNEL_TOPICS: dict[str, list] = {
    # Philosophy
    "PhilosophyTube":       ["philosophy", "politics"],
    "Philosophize This!":   ["philosophy"],
    "The School of Life":   ["philosophy", "psychology"],
    "einzelgänger":         ["philosophy", "stoicism"],
    "Wireless Philosophy":  ["philosophy"],
    "Kane B":               ["philosophy", "analytic"],
    "Alex O'Connor":        ["philosophy", "atheism"],

    # AI / ML
    "Andrej Karpathy":      ["ai", "ml", "deeplearning"],
    "Yannic Kilcher":       ["ai", "ml", "papers"],
    "Lex Fridman":          ["ai", "philosophy", "interviews"],
    "Two Minute Papers":    ["ai", "research"],
    "Computerphile":        ["cs", "ai"],
    "3Blue1Brown":          ["maths", "ml", "visualization"],
    "StatQuest":            ["ml", "statistics"],
    "sentdex":              ["ml", "python"],
    "Sebastian Lague":      ["cs", "graphics"],

    # Maths
    "Numberphile":          ["maths"],
    "Mathologer":           ["maths"],
    "Veritasium":           ["science", "maths", "physics"],
    "Stand-up Maths":       ["maths"],

    # Physics / Science
    "PBS Space Time":       ["physics", "cosmology"],
    "SciShow":              ["science"],
    "Kurzgesagt":           ["science", "philosophy"],
    "Sean Carroll":         ["physics", "philosophy"],
    "Scott Manley":         ["space", "physics"],

    # CS / Programming
    "Fireship":             ["cs", "webdev"],
    "The Primeagen":        ["cs", "rust", "programming"],
    "ThePrimeTime":         ["cs", "programming"],
    "George Hotz":          ["cs", "ai", "hacking"],
    "LiveOverflow":         ["security", "hacking"],

    # Economics / Social
    "Economics Explained":  ["economics"],
    "Patrick Boyle":        ["finance", "economics"],

    # Language / Linguistics
    "Tom Scott":            ["language", "culture", "science"],
    "NativLang":            ["linguistics"],

    # Self-development
    "Andrew Huberman":      ["neuroscience", "health"],
    "Tim Ferriss":          ["productivity", "interviews"],
}

# Keyword heuristics for title-based topic inference
_TITLE_TOPIC_PATTERNS = [
    (r'\b(philosophy|philosopher|hegel|kant|nietzsche|wittgenstein|plato|aristotle|spinoza|'
     r'phenomenology|epistem|ontolog|metaphysic|ethics|consciousness)\b', "philosophy"),
    (r'\b(machine learning|neural network|deep learning|transformer|llm|gpt|claude|'
     r'ai|artificial intelligence|diffusion model|reinforcement)\b', "ai"),
    (r'\b(math|calculus|linear algebra|topology|number theory|proof|theorem|'
     r'abstract algebra|category theory|statistics)\b', "maths"),
    (r'\b(physics|quantum|relativity|particle|cosmology|entropy|thermodynamic|'
     r'electro|spacetime)\b', "physics"),
    (r'\b(programming|python|rust|javascript|algorithm|data structure|'
     r'compiler|operating system|kernel|software)\b', "cs"),
    (r'\b(psychology|cognitive|neuroscience|behavior|freud|jung|motivation)\b', "psychology"),
    (r'\b(history|ancient|medieval|war|empire|revolution|civilization)\b', "history"),
    (r'\b(economics|market|finance|investment|capitalism|inflation|gdp)\b', "economics"),
    (r'\b(music|guitar|piano|composition|theory|chord|melody)\b', "music"),
    (r'\b(language|linguistics|grammar|syntax|etymology|translation)\b', "linguistics"),
    (r'\b(security|hacking|ctf|exploit|vulnerability|reverse engineering|malware)\b', "security"),
    (r'\b(space|nasa|rocket|satellite|mars|moon|astronomy|telescope)\b', "space"),
]


def _categorize(title: str, channel: str) -> list:
    """Return topic list for a video given its title and channel name."""
    topics = []

    # Channel lookup first (most reliable)
    for ch_key, ch_topics in _CHANNEL_TOPICS.items():
        if ch_key.lower() in channel.lower() or channel.lower() in ch_key.lower():
            topics.extend(ch_topics)
            break

    # Title keyword fallback (or supplement)
    title_lower = title.lower()
    for pattern, topic in _TITLE_TOPIC_PATTERNS:
        if re.search(pattern, title_lower) and topic not in topics:
            topics.append(topic)

    return topics or ["general"]


# ─── Main analyzer ────────────────────────────────────────────────────────────

class YouTubeAnalyzer:
    def __init__(self, cfg: dict, binge_gap_minutes: int = 30,
                 synergy_min_lag_weeks: int = 2,
                 synergy_max_lag_weeks: int = 24):
        self.cfg = cfg
        self.binge_gap      = timedelta(minutes=binge_gap_minutes)
        self.synergy_min    = timedelta(weeks=synergy_min_lag_weeks)
        self.synergy_max    = timedelta(weeks=synergy_max_lag_weeks)

    def analyze(self,
                watch_path:   Path,
                search_path:  Optional[Path] = None,
                playlist_dir: Optional[Path] = None) -> "YouTubePersonaReport":

        log.info("Loading watch history …")
        watches = self._load_watches(watch_path)
        log.info(f"  {len(watches):,} watch events loaded")

        searches = []
        if search_path and Path(search_path).exists():
            searches = self._load_searches(Path(search_path))
            log.info(f"  {len(searches):,} search events loaded")

        playlists = []
        if playlist_dir and Path(playlist_dir).exists():
            playlists = self._load_playlists(Path(playlist_dir))
            log.info(f"  {len(playlists):,} playlist saves loaded")

        log.info("Categorizing topics …")
        for v in watches:
            v.topics = _categorize(v.title, v.channel)
        for s in searches:
            s.topics = _infer_search_topics(s.query)

        log.info("Running analysis passes …")
        timeline     = self._temporal_persona_timeline(watches)
        drift        = self._topic_drift(timeline)
        synergy      = self._algorithm_search_synergy(watches, searches)
        time_alloc   = self._time_allocation(watches)
        binge_report = self._binge_sessions(watches)
        playlist_rep = self._playlist_crystallization(watches, playlists) if playlists else {}

        log.info("LLM characterization …")
        llm_summary = self._llm_characterize(timeline, drift, synergy, watches)

        report = YouTubePersonaReport(
            total_videos    = len(watches),
            total_searches  = len(searches),
            date_range      = self._date_range(watches),
            timeline        = timeline,
            drift_events    = drift,
            synergy_signals = synergy,
            time_allocation = time_alloc,
            binge_report    = binge_report,
            playlist_report = playlist_rep,
            llm_summary     = llm_summary,
        )
        return report

    # ── 1. Temporal persona timeline ──────────────────────────────────────────

    def _temporal_persona_timeline(self, watches: list[VideoEvent]) -> list[TopicWindow]:
        monthly: dict[str, list[VideoEvent]] = defaultdict(list)
        for v in watches:
            monthly[v.month].append(v)

        windows = []
        for month in sorted(monthly.keys()):
            vids = monthly[month]
            days_in_month = 30  # approximation

            channel_counts = Counter(v.channel for v in vids if v.channel)
            topic_counts   = Counter(t for v in vids for t in v.topics)

            # Diversity: 1 - sum(p^2) (Herfindahl index complement)
            total = max(len(vids), 1)
            ch_probs = [c/total for c in channel_counts.values()]
            diversity = 1 - sum(p**2 for p in ch_probs)

            # Time-of-day profile
            tod = {"morning": 0, "afternoon": 0, "evening": 0, "night": 0}
            for v in vids:
                h = v.hour
                if 5 <= h < 12:   tod["morning"]   += 1
                elif 12 <= h < 17: tod["afternoon"] += 1
                elif 17 <= h < 22: tod["evening"]   += 1
                else:              tod["night"]      += 1
            tod_pct = {k: round(c/total, 3) for k, c in tod.items()}

            weekend_count = sum(1 for v in vids if v.weekday >= 5)

            windows.append(TopicWindow(
                month           = month,
                video_count     = len(vids),
                top_channels    = channel_counts.most_common(5),
                top_topics      = topic_counts.most_common(5),
                watch_velocity  = round(len(vids) / days_in_month, 2),
                diversity_score = round(diversity, 3),
                tod_profile     = tod_pct,
                weekend_fraction= round(weekend_count / total, 3),
            ))

        return windows

    # ── 2. Topic drift detection ──────────────────────────────────────────────

    def _topic_drift(self, timeline: list[TopicWindow],
                     window_size: int = 3) -> list[DriftEvent]:
        """
        Sliding window comparison: compare topic distribution of current window
        to previous window. Large divergence = drift event.
        """
        if len(timeline) < window_size * 2:
            return []

        events = []
        for i in range(window_size, len(timeline)):
            prev_window  = timeline[i-window_size:i]
            curr_window  = timeline[i:i+window_size]
            if i + window_size > len(timeline):
                break

            prev_topics = Counter()
            for w in prev_window:
                for t, c in w.top_topics:
                    prev_topics[t] += c

            curr_topics = Counter()
            for w in curr_window:
                for t, c in w.top_topics:
                    curr_topics[t] += c

            # Normalize
            pt = sum(prev_topics.values()) or 1
            ct = sum(curr_topics.values()) or 1
            prev_dist = {t: c/pt for t, c in prev_topics.items()}
            curr_dist = {t: c/ct for t, c in curr_topics.items()}

            all_topics = set(prev_dist) | set(curr_dist)
            # Jensen-Shannon-ish distance
            magnitude = sum(
                abs(curr_dist.get(t, 0) - prev_dist.get(t, 0))
                for t in all_topics
            ) / max(len(all_topics), 1)

            if magnitude > 0.12:  # threshold for meaningful drift
                gained = [t for t in curr_dist if curr_dist[t] > prev_dist.get(t, 0) + 0.05]
                lost   = [t for t in prev_dist if prev_dist[t] > curr_dist.get(t, 0) + 0.05]
                desc = f"Viewing shifted: gained [{', '.join(gained[:3])}]"
                if lost:
                    desc += f", reduced [{', '.join(lost[:3])}]"

                events.append(DriftEvent(
                    month         = timeline[i].month,
                    description   = desc,
                    topics_gained = gained,
                    topics_lost   = lost,
                    magnitude     = round(magnitude, 3),
                ))

        # Deduplicate close drift events (keep the strongest per 2-month window)
        return _deduplicate_drift(events)

    # ── 3. Algorithm → Search synergy ────────────────────────────────────────

    def _algorithm_search_synergy(self,
                                   watches:  list[VideoEvent],
                                   searches: list[SearchEvent]) -> list[SynergySignal]:
        if not searches:
            return []

        # Build topic → first_watch_date mapping
        topic_first_watch: dict[str, datetime] = {}
        topic_watch_count: Counter = Counter()
        for v in sorted(watches, key=lambda x: x.timestamp):
            for t in v.topics:
                if t not in topic_first_watch:
                    topic_first_watch[t] = v.timestamp
                topic_watch_count[t] += 1

        # Build topic → first_search_date mapping
        topic_first_search: dict[str, datetime] = {}
        topic_search_count: Counter = Counter()
        for s in sorted(searches, key=lambda x: x.timestamp):
            for t in s.topics:
                if t not in topic_first_search:
                    topic_first_search[t] = s.timestamp
                topic_search_count[t] += 1

        # Find topics that were watched first, then searched
        signals = []
        all_topics = set(topic_first_watch) & set(topic_first_search)
        for topic in all_topics:
            watch_ts  = topic_first_watch[topic]
            search_ts = topic_first_search[topic]
            lag = search_ts - watch_ts

            if self.synergy_min <= lag <= self.synergy_max:
                signals.append(SynergySignal(
                    topic         = topic,
                    first_watched = watch_ts.strftime("%Y-%m"),
                    first_searched= search_ts.strftime("%Y-%m"),
                    lag_weeks     = round(lag.days / 7, 1),
                    watch_count   = topic_watch_count[topic],
                    search_count  = topic_search_count[topic],
                ))

        # Sort by strongest evidence (high watch count before the search)
        signals.sort(key=lambda s: s.watch_count, reverse=True)
        return signals

    # ── 4. Time allocation analysis ───────────────────────────────────────────

    def _time_allocation(self, watches: list[VideoEvent]) -> dict:
        if not watches:
            return {}

        # Yearly stats
        yearly: dict[int, list] = defaultdict(list)
        for v in watches:
            yearly[v.timestamp.year].append(v)

        yearly_stats = {}
        for yr, vids in sorted(yearly.items()):
            topic_c = Counter(t for v in vids for t in v.topics)
            ch_c    = Counter(v.channel for v in vids if v.channel)
            yearly_stats[str(yr)] = {
                "video_count":     len(vids),
                "top_topics":      topic_c.most_common(5),
                "top_channels":    ch_c.most_common(5),
                "avg_per_month":   round(len(vids) / 12, 1),
            }

        # Time-of-day distribution overall
        tod_overall = Counter()
        for v in watches:
            h = v.hour
            if 5 <= h < 12:    tod_overall["morning"]   += 1
            elif 12 <= h < 17: tod_overall["afternoon"] += 1
            elif 17 <= h < 22: tod_overall["evening"]   += 1
            else:              tod_overall["night"]      += 1

        total = max(len(watches), 1)
        tod_pct = {k: round(c/total*100, 1) for k, c in tod_overall.items()}

        # Weekday vs weekend
        weekday_count = sum(1 for v in watches if v.weekday < 5)
        weekend_count = total - weekday_count

        # Most productive day of week
        dow = Counter(v.weekday for v in watches)
        dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        dow_dist = {dow_names[d]: c for d, c in dow.most_common()}

        # Velocity trend: compare first half vs second half of history
        sorted_watches = sorted(watches, key=lambda v: v.timestamp)
        mid = len(sorted_watches) // 2
        first_half = sorted_watches[:mid]
        second_half= sorted_watches[mid:]
        first_days  = max((first_half[-1].timestamp - first_half[0].timestamp).days, 1)
        second_days = max((second_half[-1].timestamp - second_half[0].timestamp).days, 1)
        velocity_trend = {
            "first_half_vpd":  round(len(first_half)  / first_days,  2),
            "second_half_vpd": round(len(second_half) / second_days, 2),
        }
        if velocity_trend["second_half_vpd"] > velocity_trend["first_half_vpd"] * 1.2:
            velocity_trend["trend"] = "accelerating"
        elif velocity_trend["second_half_vpd"] < velocity_trend["first_half_vpd"] * 0.8:
            velocity_trend["trend"] = "decelerating"
        else:
            velocity_trend["trend"] = "stable"

        return {
            "yearly":          yearly_stats,
            "time_of_day":     tod_pct,
            "weekday_count":   weekday_count,
            "weekend_count":   weekend_count,
            "day_of_week":     dow_dist,
            "velocity_trend":  velocity_trend,
            "peak_hour":       max(Counter(v.hour for v in watches).items(),
                                   key=lambda x: x[1])[0],
        }

    # ── 5. Binge session detection ────────────────────────────────────────────

    def _binge_sessions(self, watches: list[VideoEvent],
                        min_videos: int = 5) -> dict:
        """Detect sessions where many videos were watched in rapid succession."""
        if not watches:
            return {}

        sorted_w = sorted(watches, key=lambda v: v.timestamp)
        sessions: list[BingeSession] = []
        current: list[VideoEvent] = [sorted_w[0]]

        for v in sorted_w[1:]:
            if v.timestamp - current[-1].timestamp <= self.binge_gap:
                current.append(v)
            else:
                if len(current) >= min_videos:
                    topic_c = Counter(t for vid in current for t in vid.topics)
                    sessions.append(BingeSession(
                        start  = current[0].timestamp,
                        end    = current[-1].timestamp,
                        videos = [vid.title for vid in current],
                        topics = [t for t, _ in topic_c.most_common(3)],
                    ))
                current = [v]

        # Binge stats
        if not sessions:
            return {"session_count": 0}

        avg_duration = sum(s.duration_minutes for s in sessions) / len(sessions)
        avg_videos   = sum(s.video_count for s in sessions) / len(sessions)
        topic_binge  = Counter(t for s in sessions for t in s.topics)

        top_sessions = sorted(sessions, key=lambda s: s.video_count, reverse=True)[:5]
        return {
            "session_count":    len(sessions),
            "avg_duration_min": round(avg_duration, 1),
            "avg_videos":       round(avg_videos, 1),
            "most_binged_topics": topic_binge.most_common(5),
            "top_sessions": [
                {
                    "date":    s.start.strftime("%Y-%m-%d"),
                    "videos":  s.video_count,
                    "minutes": round(s.duration_minutes, 0),
                    "topics":  s.topics,
                }
                for s in top_sessions
            ],
        }

    # ── 6. Playlist crystallization ───────────────────────────────────────────

    def _playlist_crystallization(self,
                                   watches:   list[VideoEvent],
                                   playlists: list[PlaylistEvent]) -> dict:
        """
        For each topic in playlists: how long between first watching any video
        on that topic and the first save to a playlist?
        """
        watch_by_id = {v.video_id: v for v in watches}
        topic_first_watch: dict[str, datetime] = {}
        for v in sorted(watches, key=lambda x: x.timestamp):
            for t in v.topics:
                if t not in topic_first_watch:
                    topic_first_watch[t] = v.timestamp

        lags: list[dict] = []
        for pe in playlists:
            if not pe.added_at:
                continue
            watched = watch_by_id.get(pe.video_id)
            if not watched:
                continue
            lag_days = (pe.added_at - watched.timestamp).days
            if lag_days < 0:
                continue
            lags.append({
                "playlist": pe.playlist_name,
                "title":    pe.title,
                "lag_days": lag_days,
                "topics":   watched.topics,
            })

        if not lags:
            return {}

        avg_lag = sum(l["lag_days"] for l in lags) / len(lags)
        instant = [l for l in lags if l["lag_days"] <= 1]
        slow    = [l for l in lags if l["lag_days"] > 30]

        return {
            "total_saves":      len(lags),
            "avg_lag_days":     round(avg_lag, 1),
            "instant_saves":    len(instant),      # same day
            "slow_burn_saves":  len(slow),          # >30 days
            "examples": {
                "instant":  [l["title"] for l in instant[:3]],
                "slow_burn":[l["title"] for l in slow[:3]],
            },
        }

    # ── LLM characterization ─────────────────────────────────────────────────

    def _llm_characterize(self, timeline: list[TopicWindow],
                           drift: list[DriftEvent],
                           synergy: list[SynergySignal],
                           watches: list[VideoEvent]) -> str:
        if not timeline:
            return ""

        # Build a compact summary for the prompt
        monthly_summary = []
        for w in timeline[-24:]:  # last 2 years
            top_t = ", ".join(t for t, _ in w.top_topics[:3])
            monthly_summary.append(f"  {w.month}: {w.video_count} videos  [{top_t}]  velocity={w.watch_velocity}/day")

        drift_summary = "\n".join(
            f"  {d.month}: {d.description} (magnitude={d.magnitude})"
            for d in drift[:5]
        )

        synergy_summary = "\n".join(
            f"  {s.topic}: watched from {s.first_watched}, searched {s.lag_weeks:.0f}wks later"
            for s in synergy[:5]
        )

        total_years = (watches[-1].timestamp - watches[0].timestamp).days / 365 if len(watches) > 1 else 1
        all_topics = Counter(t for v in watches for t in v.topics)

        prompt = f"""You are analysing someone's YouTube viewing history spanning {total_years:.1f} years
({len(watches):,} videos total).

Top topics overall: {', '.join(t for t, _ in all_topics.most_common(8))}

Recent monthly viewing (last 2 years, newest last):
{chr(10).join(monthly_summary[-12:])}

Major interest shifts detected:
{drift_summary or '  (none detected)'}

Algorithm-to-search synergy signals (topics passively watched before actively searched):
{synergy_summary or '  (none detected)'}

Write a 150-word portrait of this person's intellectual journey as seen through
their YouTube history. Focus on:
1. What drove them at the start vs now
2. The most significant shift in interests
3. What the algorithm→search signals reveal about how they absorb new ideas
4. One sentence on their viewing rhythm (when and how intensely they watch)

Write in second person ("Your YouTube history reveals..."). Be specific."""

        try:
            return self._llm_call(prompt, max_tokens=350)
        except Exception as e:
            log.warning(f"LLM characterization failed: {e}")
            top5 = [t for t, _ in all_topics.most_common(5)]
            return (f"YouTube history spans {total_years:.1f} years, "
                    f"{len(watches):,} videos. Top topics: {', '.join(top5)}.")

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_watches(self, path: Path) -> list[VideoEvent]:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        events = []
        for item in data:
            # Filter: only actual watch events (not ads, not "Watched a video")
            title_raw = item.get("title", "")
            if not title_raw.startswith("Watched "):
                continue
            title = title_raw[len("Watched "):].strip()

            url = item.get("titleUrl", "")
            video_id = ""
            m = re.search(r'[?&]v=([A-Za-z0-9_-]{11})', url)
            if m:
                video_id = m.group(1)

            channel = ""
            subs = item.get("subtitles", [])
            if subs and isinstance(subs, list) and isinstance(subs[0], dict):
                channel = subs[0].get("name", "")

            ts_str = item.get("time", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                continue

            events.append(VideoEvent(
                title=title, channel=channel, url=url,
                video_id=video_id, timestamp=ts,
            ))

        return sorted(events, key=lambda v: v.timestamp)

    def _load_searches(self, path: Path) -> list[SearchEvent]:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        events = []
        for item in data:
            hdr = item.get("header", "")
            if hdr not in ("YouTube", "YouTube Search"):
                continue
            title = item.get("title", "")
            query = title.replace("Searched for ", "").strip() if title.startswith("Searched") else title
            if not query:
                continue
            ts_str = item.get("time", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                continue
            events.append(SearchEvent(query=query, timestamp=ts))

        return sorted(events, key=lambda s: s.timestamp)

    def _load_playlists(self, playlist_dir: Path) -> list[PlaylistEvent]:
        events = []
        for jf in Path(playlist_dir).glob("*.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
                name = data.get("title", jf.stem)
                for item in data.get("playlistItems", data.get("videos", [])):
                    vid_id = (item.get("contentDetails", {}).get("videoId")
                              or item.get("videoId", ""))
                    title  = (item.get("snippet", {}).get("title")
                              or item.get("title", ""))
                    added  = (item.get("snippet", {}).get("publishedAt")
                              or item.get("addedAt") or item.get("created", ""))
                    ts = None
                    if added:
                        try:
                            ts = datetime.fromisoformat(added.replace("Z", "+00:00"))
                        except Exception:
                            pass
                    if vid_id:
                        events.append(PlaylistEvent(
                            playlist_name=name, video_id=vid_id,
                            title=title, added_at=ts
                        ))
            except Exception as e:
                log.debug(f"Playlist parse failed for {jf.name}: {e}")
        return events

    def _date_range(self, watches: list[VideoEvent]) -> dict:
        if not watches:
            return {}
        s = sorted(watches, key=lambda v: v.timestamp)
        return {
            "first": s[0].timestamp.strftime("%Y-%m-%d"),
            "last":  s[-1].timestamp.strftime("%Y-%m-%d"),
            "span_days": (s[-1].timestamp - s[0].timestamp).days,
        }

    def _llm_call(self, prompt: str, max_tokens: int = 512) -> str:
        backend = self.cfg.get("llm_backend", "claude")
        if backend == "ollama":
            base    = self.cfg.get("ollama_base_url", "http://localhost:11434").rstrip("/")
            model   = self.cfg.get("ollama_model", "mistral")
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


# ─── Report ───────────────────────────────────────────────────────────────────

@dataclass
class YouTubePersonaReport:
    total_videos:    int
    total_searches:  int
    date_range:      dict
    timeline:        list[TopicWindow]
    drift_events:    list[DriftEvent]
    synergy_signals: list[SynergySignal]
    time_allocation: dict
    binge_report:    dict
    playlist_report: dict
    llm_summary:     str

    def print_summary(self):
        dr = self.date_range
        print(f"\n{'═'*65}")
        print(f"  YOUTUBE PERSONA REPORT")
        print(f"  {dr.get('first','?')} → {dr.get('last','?')}"
              f"  ({dr.get('span_days',0)//365} years)")
        print(f"{'═'*65}")
        print(f"  {self.total_videos:,} videos  |  {self.total_searches:,} searches\n")

        if self.llm_summary:
            print(f"  {self.llm_summary}\n")

        print(f"── TEMPORAL PERSONA TIMELINE ({'last 12 months shown'}) ───────────")
        for w in self.timeline[-12:]:
            top_t = " | ".join(f"{t}({c})" for t, c in w.top_topics[:3])
            bar   = "▓" * min(w.video_count // 5, 30)
            print(f"  {w.month}  {bar:30s}  {w.video_count:3d}v  [{top_t}]")

        if self.drift_events:
            print(f"\n── TOPIC DRIFT EVENTS ({len(self.drift_events)} detected) ───────────────────")
            for d in self.drift_events:
                strength = "●●●" if d.magnitude > 0.25 else "●●" if d.magnitude > 0.15 else "●"
                print(f"  {d.month}  {strength}  {d.description}")

        if self.synergy_signals:
            print(f"\n── ALGORITHM → SEARCH SYNERGY ({len(self.synergy_signals)} signals) ─────────────")
            print(f"  (Topics you passively watched, then started actively searching)\n")
            for s in self.synergy_signals[:8]:
                arrow = f"{s.first_watched} →{'─'*int(s.lag_weeks//2)}→ {s.first_searched}"
                print(f"  [{s.topic:<15}]  {arrow}  lag={s.lag_weeks:.0f}wks")

        ta = self.time_allocation
        if ta:
            print(f"\n── TIME ALLOCATION ──────────────────────────────────────────────")
            tod = ta.get("time_of_day", {})
            print(f"  Time of day:  morning {tod.get('morning',0):.0f}%  "
                  f"afternoon {tod.get('afternoon',0):.0f}%  "
                  f"evening {tod.get('evening',0):.0f}%  "
                  f"night {tod.get('night',0):.0f}%")
            print(f"  Peak hour:    {ta.get('peak_hour','?')}:00")
            wkd = ta.get("weekday_count",0)
            wke = ta.get("weekend_count",0)
            tot = max(wkd + wke, 1)
            print(f"  Weekday:      {wkd/tot*100:.0f}%  Weekend: {wke/tot*100:.0f}%")
            vt = ta.get("velocity_trend", {})
            print(f"  Velocity:     {vt.get('first_half_vpd','?')} → "
                  f"{vt.get('second_half_vpd','?')} v/day  ({vt.get('trend','?')})")

        br = self.binge_report
        if br and br.get("session_count", 0) > 0:
            print(f"\n── BINGE SESSIONS ({br['session_count']} detected) ────────────────────────")
            print(f"  Avg: {br['avg_videos']:.0f} videos over {br['avg_duration_min']:.0f}min")
            print(f"  Most binged topics: "
                  f"{', '.join(t for t,_ in br.get('most_binged_topics',[])[:4])}")

        if self.playlist_report:
            pr = self.playlist_report
            print(f"\n── PLAYLIST CRYSTALLIZATION ─────────────────────────────────────")
            print(f"  {pr.get('total_saves',0)} saves  |  avg lag: {pr.get('avg_lag_days',0):.0f} days")
            print(f"  Instant (<1d): {pr.get('instant_saves',0)}  "
                  f"Slow burn (>30d): {pr.get('slow_burn_saves',0)}")

        print(f"{'═'*65}\n")

    def to_dict(self) -> dict:
        return {
            "total_videos":   self.total_videos,
            "total_searches": self.total_searches,
            "date_range":     self.date_range,
            "llm_summary":    self.llm_summary,
            "timeline": [
                {
                    "month":          w.month,
                    "video_count":    w.video_count,
                    "top_channels":   w.top_channels,
                    "top_topics":     w.top_topics,
                    "watch_velocity": w.watch_velocity,
                    "diversity":      w.diversity_score,
                    "time_of_day":    w.tod_profile,
                    "weekend_frac":   w.weekend_fraction,
                }
                for w in self.timeline
            ],
            "drift_events": [
                {
                    "month":         d.month,
                    "description":   d.description,
                    "topics_gained": d.topics_gained,
                    "topics_lost":   d.topics_lost,
                    "magnitude":     d.magnitude,
                }
                for d in self.drift_events
            ],
            "synergy_signals": [
                {
                    "topic":         s.topic,
                    "first_watched": s.first_watched,
                    "first_searched":s.first_searched,
                    "lag_weeks":     s.lag_weeks,
                    "watch_count":   s.watch_count,
                    "search_count":  s.search_count,
                }
                for s in self.synergy_signals
            ],
            "time_allocation": self.time_allocation,
            "binge_report":    self.binge_report,
            "playlist_report": self.playlist_report,
        }

    def save(self, path: str | Path):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        log.info(f"YouTube report saved to {out}")

    def integrate_with_persona(self, persona_path: str | Path = "data/persona.json"):
        """Merge the YouTube timeline into the main persona.json file."""
        p = Path(persona_path)
        if not p.exists():
            log.warning(f"Persona file not found at {p}. Run persona build first.")
            return

        profile = json.loads(p.read_text())
        profile["youtube_arc"] = {
            "generated_at":   datetime.now(timezone.utc).isoformat(),
            "total_videos":   self.total_videos,
            "date_range":     self.date_range,
            "timeline":       self.to_dict()["timeline"],
            "drift_events":   self.to_dict()["drift_events"],
            "synergy_signals":self.to_dict()["synergy_signals"],
            "time_allocation":self.time_allocation,
            "llm_summary":    self.llm_summary,
        }
        p.write_text(json.dumps(profile, indent=2, default=str))
        log.info(f"Persona updated with YouTube arc at {p}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _infer_search_topics(query: str) -> list:
    """Infer topic tags from a search query string."""
    q = query.lower()
    topics = []
    for pattern, topic in _TITLE_TOPIC_PATTERNS:
        if re.search(pattern, q):
            topics.append(topic)
    return topics or ["general"]


def _deduplicate_drift(events: list[DriftEvent]) -> list[DriftEvent]:
    """Keep only the strongest drift event per 2-month window."""
    if not events:
        return []
    result = [events[0]]
    for e in events[1:]:
        last = result[-1]
        last_ym = tuple(int(x) for x in last.month.split("-"))
        curr_ym = tuple(int(x) for x in e.month.split("-"))
        months_apart = (curr_ym[0] - last_ym[0]) * 12 + (curr_ym[1] - last_ym[1])
        if months_apart < 2:
            if e.magnitude > last.magnitude:
                result[-1] = e
        else:
            result.append(e)
    return result
