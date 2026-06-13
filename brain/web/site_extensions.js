/* ============================================================
   SITE EXTENSIONS
   Projects filtering · Writing grid · Brain pulse
   ============================================================ */

'use strict';

(function () {

  /* ── Project Category Filters ──────────────────────────── */
  const projFilters = document.querySelectorAll('.proj-filter');
  const projCards   = document.querySelectorAll('.proj-card');

  projFilters.forEach(btn => {
    btn.addEventListener('click', () => {
      projFilters.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const cat = btn.dataset.cat;
      let delay = 0;

      projCards.forEach(card => {
        if (cat === 'all' || card.dataset.cat === cat) {
          card.hidden = false;
          card.style.animationDelay = delay + 'ms';
          delay += 60;
        } else {
          card.hidden = true;
        }
      });
    });
  });

  /* ── Writing Grid (from writing.json) ──────────────────── */
  async function loadWriting() {
    const grid  = document.getElementById('writing-grid');
    const empty = document.getElementById('writing-empty');
    if (!grid) return;

    try {
      const res = await fetch('writing.json');
      if (!res.ok) throw new Error('no writing.json');
      const pieces = await res.json();

      if (!pieces || !pieces.length) {
        if (empty) empty.style.display = '';
        return;
      }

      grid.innerHTML = '';
      pieces.forEach((piece, i) => {
        const card = document.createElement('div');
        card.className = 'writing-card';
        card.style.animationDelay = (i * 50) + 'ms';
        card.innerHTML = `
          ${piece.tag ? `<div class="writing-card-tag">${esc(piece.tag)}</div>` : ''}
          <div class="writing-card-title">${esc(piece.title || 'Untitled')}</div>
          <div class="writing-card-excerpt">${esc(piece.excerpt || '')}</div>
          <div class="writing-card-meta">${
            [piece.date, piece.word_count ? piece.word_count.toLocaleString() + ' words' : null]
              .filter(Boolean).join(' · ')
          }</div>
        `;

        // Open in reader if content exists, else link out
        if (piece.content) {
          card.addEventListener('click', () => {
            const reader = document.getElementById('note-reader');
            const titleEl = document.getElementById('reader-title');
            const bodyEl  = document.getElementById('reader-body');
            const metaEl  = document.getElementById('reader-meta');
            if (!reader) return;
            titleEl.textContent = piece.title || 'Untitled';
            metaEl.textContent = [piece.tag, piece.date].filter(Boolean).join(' · ');
            bodyEl.innerHTML = simpleRender(piece.content);
            reader.hidden = false;
            document.body.style.overflow = 'hidden';
          });
        } else if (piece.url) {
          card.addEventListener('click', () => window.open(piece.url, '_blank'));
        }

        grid.appendChild(card);
      });
    } catch {
      // No writing.json — show the empty-state message
      if (empty) empty.style.display = '';
    }
  }

  /* ── Brain Pulse (About sidebar stats) ─────────────────── */
  async function loadBrainPulse() {
    const el = document.getElementById('brain-pulse');
    if (!el) return;

    // Try to load persona.json for stats
    try {
      const res = await fetch('persona.json');
      if (!res.ok) throw new Error();
      const p = await res.json();

      const stats = [];

      if (p.topical_fingerprint?.top_tags) {
        const topTag = Object.entries(p.topical_fingerprint.top_tags)
          .sort((a, b) => b[1] - a[1])[0];
        if (topTag) stats.push({ label: 'Top topic', val: topTag[0] });
        const tagCount = Object.keys(p.topical_fingerprint.top_tags).length;
        stats.push({ label: 'Themes', val: String(tagCount) });
      }

      if (p.stylistic_markers) {
        if (p.stylistic_markers.avg_sentence_length) {
          stats.push({ label: 'Avg sentence', val: p.stylistic_markers.avg_sentence_length + ' words' });
        }
        if (p.stylistic_markers.vocabulary_richness) {
          stats.push({ label: 'Lexical richness', val: (p.stylistic_markers.vocabulary_richness * 100).toFixed(0) + '%' });
        }
      }

      if (p.argument_patterns) {
        const top = Object.entries(p.argument_patterns)
          .sort((a, b) => b[1] - a[1])[0];
        if (top) stats.push({ label: 'Primary logic', val: top[0].replace(/_/g, ' ') });
      }

      if (stats.length) {
        el.innerHTML = stats.map(s =>
          `<div class="brain-pulse-row">
            <span class="brain-pulse-label">${esc(s.label)}</span>
            <span class="brain-pulse-val">${esc(s.val)}</span>
          </div>`
        ).join('');
      } else {
        el.style.display = 'none';
      }
    } catch {
      // No persona data — hide the pulse widget
      el.style.display = 'none';
    }
  }

  /* ── Helpers ────────────────────────────────────────────── */
  function esc(str) {
    if (!str) return '';
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
  }

  function simpleRender(text) {
    if (!text) return '';
    return text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/^### (.+)$/gm, '<h3>$1</h3>')
      .replace(/^## (.+)$/gm, '<h2>$1</h2>')
      .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/^(?!<[h|b|p])/, '<p>')
      .replace(/$(?![>])/, '</p>');
  }

  /* ── Omni Intelligence Dashboard ─────────────────────────── */
  async function loadOmniStats() {
    const section = document.getElementById('omni-section');
    if (!section) return;

    try {
      const res = await fetch('omni_stats.json');
      if (!res.ok) throw new Error('no omni_stats.json');
      const s = await res.json();

      section.style.display = '';

      // ── Big counters ────────────────────────────────────
      const countersEl = document.getElementById('omni-counters');
      if (countersEl) {
        const items = [
          { val: fmtNum(s.corpus?.total_notes), label: 'total notes' },
          { val: fmtNum(s.corpus?.total_words), label: 'total words' },
          { val: fmtNum(s.corpus?.total_edges), label: 'edges' },
          { val: fmtNum(s.llm?.total_conversations), label: 'LLM conversations' },
          { val: fmtNum(s.llm?.user_prompts), label: 'prompts sent' },
          { val: fmtNum(s.llm?.assistant_responses), label: 'responses received' },
        ];
        countersEl.innerHTML = items.map(i =>
          `<div class="omni-counter-item">
            <span class="omni-counter-val">${i.val}</span>
            <span class="omni-counter-label">${esc(i.label)}</span>
          </div>`
        ).join('');
      }

      // ── Source composition ──────────────────────────────
      const srcBody = document.getElementById('omni-sources-body');
      if (srcBody && s.sources?.by_type) {
        const entries = Object.entries(s.sources.by_type)
          .sort((a, b) => b[1].notes - a[1].notes);
        const maxNotes = entries[0]?.[1]?.notes || 1;

        // Source type bars
        let html = entries.map(([type, data]) => {
          const pct = (data.notes / maxNotes * 100).toFixed(1);
          const label = type.replace(/_/g, ' ');
          return `<div class="omni-source-row">
            <span class="omni-source-label">${esc(label)}</span>
            <div class="omni-source-bar-bg">
              <div class="omni-source-bar" data-pct="${pct}"></div>
            </div>
            <span class="omni-source-count">${fmtNum(data.notes)}</span>
          </div>`;
        }).join('');

        // Provenance split
        if (s.sources?.provenance) {
          const out = s.sources.provenance.output;
          const inp = s.sources.provenance.input;
          if (out && inp) {
            const total = out.notes + inp.notes;
            const outPct = ((out.notes / total) * 100).toFixed(0);
            html += `<div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);">
              <div class="omni-source-row">
                <span class="omni-source-label" style="color:var(--gold)">Your words</span>
                <div class="omni-source-bar-bg">
                  <div class="omni-source-bar" data-pct="${outPct}" style="background:var(--gold)"></div>
                </div>
                <span class="omni-source-count">${outPct}%</span>
              </div>
              <div class="omni-source-row">
                <span class="omni-source-label" style="color:var(--cosmic)">External input</span>
                <div class="omni-source-bar-bg">
                  <div class="omni-source-bar" data-pct="${100-outPct}" style="background:var(--cosmic)"></div>
                </div>
                <span class="omni-source-count">${100-outPct}%</span>
              </div>
            </div>`;
          }
        }

        srcBody.innerHTML = html;

        // Animate bars
        requestAnimationFrame(() => {
          srcBody.querySelectorAll('.omni-source-bar').forEach(bar => {
            bar.style.width = bar.dataset.pct + '%';
          });
        });
      }

      // ── LLM breakdown ──────────────────────────────────
      const llmBody = document.getElementById('omni-llm-body');
      if (llmBody && s.llm) {
        const stats = [
          ['Total messages', fmtNum(s.llm.total_messages)],
          ['Conversations', fmtNum(s.llm.total_conversations)],
          ['Your prompts', fmtNum(s.llm.user_prompts)],
          ['AI responses', fmtNum(s.llm.assistant_responses)],
        ];

        // Words by role
        if (s.llm.words_by_role) {
          stats.push(['Words you wrote', fmtNum(s.llm.words_by_role.user)]);
          stats.push(['Words AI wrote', fmtNum(s.llm.words_by_role.assistant)]);
        }

        let html = stats.map(([label, val]) =>
          `<div class="omni-llm-stat">
            <span class="omni-llm-stat-label">${esc(label)}</span>
            <span class="omni-llm-stat-val">${val}</span>
          </div>`
        ).join('');

        // Platform breakdown
        if (s.llm.by_platform) {
          html += '<div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);">';
          for (const [platform, count] of Object.entries(s.llm.by_platform)) {
            html += `<div class="omni-llm-stat">
              <span class="omni-llm-stat-label" style="text-transform:capitalize">${esc(platform)}</span>
              <span class="omni-llm-stat-val">${fmtNum(count)} msgs</span>
            </div>`;
          }
          html += '</div>';
        }

        llmBody.innerHTML = html;
      }

      // ── Monthly activity chart ─────────────────────────
      const chartEl = document.getElementById('omni-monthly-chart');
      if (chartEl && s.temporal?.monthly_notes) {
        const months = Object.entries(s.temporal.monthly_notes);
        if (months.length > 0) {
          const max = Math.max(...months.map(m => m[1]));
          // Show last 24 months max to keep it readable
          const recent = months.slice(-24);
          chartEl.innerHTML = recent.map(([month, count]) => {
            const h = Math.max(2, (count / max) * 90);
            const label = month.slice(2); // YY-MM
            return `<div class="omni-bar-col" title="${month}: ${count} notes">
              <div class="omni-bar" style="height:${h}px"></div>
              <span class="omni-bar-label">${label}</span>
            </div>`;
          }).join('');
        }
      }

      // ── Tag landscape ──────────────────────────────────
      const tagsBody = document.getElementById('omni-tags-body');
      if (tagsBody && s.tags?.top_30) {
        const tags = Object.entries(s.tags.top_30);
        tagsBody.innerHTML = '<div class="omni-tag-cloud">' +
          tags.map(([tag, count]) =>
            `<span class="omni-tag">${esc(tag)}<span class="omni-tag-count">${count}</span></span>`
          ).join('') + '</div>';
      }

      // ── Clusters ───────────────────────────────────────
      const clustBody = document.getElementById('omni-clusters-body');
      if (clustBody && s.clusters?.length) {
        // Show top 8 clusters
        clustBody.innerHTML = s.clusters.slice(0, 8).map(c =>
          `<div class="omni-cluster-item">
            <div class="omni-cluster-head">
              <span class="omni-cluster-id">Cluster ${c.id}</span>
              <span class="omni-cluster-size">${c.size} notes</span>
            </div>
            <div class="omni-cluster-samples">${c.sample_titles.map(t => esc(t)).join(' · ')}</div>
          </div>`
        ).join('');
      }

      // ── Hub idea highlights ────────────────────────────
      const hlBody = document.getElementById('omni-highlights-body');
      if (hlBody && s.highlights?.length) {
        hlBody.innerHTML = s.highlights.map(h =>
          `<div class="omni-highlight">
            <div class="omni-highlight-title">${esc(h.title)}</div>
            ${h.excerpt ? `<div class="omni-highlight-excerpt">${esc(h.excerpt)}</div>` : ''}
            <div class="omni-highlight-meta">
              <span class="omni-highlight-centrality">centrality ${h.centrality.toFixed(5)}</span>
              <span>${fmtNum(h.word_count)} words</span>
              <span>${esc(h.source_type?.replace(/_/g, ' ') || '')}</span>
              ${h.tags?.length ? h.tags.map(t => `<span>${esc(t)}</span>`).join('') : ''}
            </div>
          </div>`
        ).join('');
      }

    } catch {
      // No omni_stats.json — section stays hidden, no error shown
    }
  }

  function fmtNum(n) {
    if (n == null || isNaN(n)) return '—';
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (n >= 1_000) return (n / 1_000).toFixed(1) + 'k';
    return String(n);
  }

  /* ── Init ──────────────────────────────────────────────── */
  document.addEventListener('DOMContentLoaded', () => {
    loadWriting();
    loadBrainPulse();
    loadOmniStats();
  });

})();
