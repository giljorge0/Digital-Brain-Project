/* ============================================================
   GIL JORGE — DIGITAL BRAIN
   app.js  ·  All routing, data fetching, D3 graph, UI
   ============================================================ */

'use strict';

/* ── State ─────────────────────────────────────────────────── */
const State = {
  notes: null,          // raw notes.json
  graph: null,          // raw graph_data.json
  persona: null,        // raw persona.json
  activeView: 'about',
  gardenCategory: 'all',
  gardenQuery: '',
  graphFrozen: true,
  graphMineOnly: false,
  graphSim: null,
  graphZoom: null,
  graphClusterMode: false,
  graphData: null,      // processed { nodes, links }
};

/* ── Helpers ───────────────────────────────────────────────── */
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function formatNumber(n) {
  if (n == null || isNaN(n)) return '—';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'k';
  return String(n);
}

function formatDate(str) {
  if (!str) return '';
  try {
    return new Date(str).toLocaleDateString('en-GB', {
      year: 'numeric', month: 'short', day: 'numeric'
    });
  } catch { return str; }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

/* Simple markdown-ish renderer for note content */
function renderMarkdown(text) {
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

/* Animate counter from 0 to target */
function animateCounter(el, target, duration = 1200) {
  if (!el || isNaN(target)) return;
  const start = performance.now();
  const step = (now) => {
    const p = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - p, 3); // ease-out-cubic
    el.textContent = formatNumber(Math.round(ease * target));
    if (p < 1) requestAnimationFrame(step);
    else el.textContent = formatNumber(target);
  };
  requestAnimationFrame(step);
}

/* ── Data Fetching ─────────────────────────────────────────── */
async function fetchJSON(path) {
  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    console.warn(`[DigitalBrain] Could not load ${path}:`, e.message);
    return null;
  }
}

/* Generate plausible demo data when JSON files are absent */
function makeDemoNotes() {
  const categories = {
    'AI': [
      { id: 'ai-1', title: 'Neuro-symbolic reasoning', content: 'An exploration of hybrid systems that combine the pattern recognition of neural networks with the explicit rule-following of symbolic AI...\n\nThe key challenge is grounding — how do symbols acquire meaning from continuous representations?', tags: ['AI', 'neuro-symbolic', 'cognition'], date: '2024-11-15', word_count: 420 },
      { id: 'ai-2', title: 'Transformer attention as memory retrieval', content: 'Attention mechanisms can be reframed as a differentiable memory system. The query-key-value triplet mirrors classical associative memory models from cognitive science.', tags: ['AI', 'transformers', 'memory'], date: '2024-10-22', word_count: 310 },
      { id: 'ai-3', title: 'Epistemic humility in language models', content: 'What does it mean for a model to "know" something? Calibration, uncertainty quantification, and the philosophy of machine epistemics.', tags: ['AI', 'epistemics', 'philosophy'], date: '2024-09-10', word_count: 580 },
    ],
    'Philosophy': [
      { id: 'ph-1', title: 'Extended mind thesis', content: 'Andy Clark and David Chalmers argue that cognition extends beyond the skull. If my notebook scaffolds my thinking, is it part of my mind?', tags: ['philosophy', 'cognition', 'mind'], date: '2024-10-05', word_count: 700 },
      { id: 'ph-2', title: 'Qualia and the hard problem', content: 'Chalmers\' distinction between easy and hard problems of consciousness. The explanatory gap that functional accounts cannot bridge.', tags: ['philosophy', 'consciousness', 'qualia'], date: '2024-08-30', word_count: 450 },
    ],
    'Engineering': [
      { id: 'en-1', title: 'Knowledge graph architectures', content: 'Survey of graph database approaches: RDF triple stores, property graphs, hypergraphs. Use cases, trade-offs, and the path to semantic reasoning.', tags: ['engineering', 'graphs', 'data'], date: '2024-11-01', word_count: 620 },
      { id: 'en-2', title: 'Retrieval-augmented generation patterns', content: 'RAG fundamentally changes how we think about model memory. Dense retrieval, sparse retrieval, and hybrid approaches compared.', tags: ['engineering', 'RAG', 'LLM'], date: '2024-09-25', word_count: 390 },
    ],
    'Reading': [
      { id: 'rd-1', title: 'Notes on Gödel, Escher, Bach', content: 'Hofstadter\'s masterwork on self-reference and strange loops. The isomorphism between formal systems and consciousness runs throughout.', tags: ['reading', 'Hofstadter', 'math'], date: '2024-07-12', word_count: 520 },
      { id: 'rd-2', title: 'Surfaces and Essences — Hofstadter & Sander', content: 'Analogies are the fuel and fire of thought. Every act of categorisation is an analogy-making act at some level of abstraction.', tags: ['reading', 'cognition', 'analogy'], date: '2024-06-18', word_count: 330 },
    ],
  };
  const total = Object.values(categories).reduce((s, a) => s + a.length, 0);
  return { total, categories };
}

function makeDemoGraph() {
  const nodes = [
    { id: 'n1', label: 'Neuro-symbolic AI', type: 'mine', group: 'AI' },
    { id: 'n2', label: 'Transformers', type: 'mine', group: 'AI' },
    { id: 'n3', label: 'Memory architectures', type: 'mine', group: 'AI' },
    { id: 'n4', label: 'Extended mind', type: 'mine', group: 'Philosophy' },
    { id: 'n5', label: 'Consciousness', type: 'mine', group: 'Philosophy' },
    { id: 'n6', label: 'Knowledge graphs', type: 'mine', group: 'Engineering' },
    { id: 'n7', label: 'RAG systems', type: 'mine', group: 'Engineering' },
    { id: 'n8', label: 'Hofstadter', type: 'mine', group: 'Reading' },
    { id: 'n9', label: 'Andy Clark', type: 'external' },
    { id: 'n10', label: 'Chalmers', type: 'external' },
    { id: 'n11', label: 'Attention mechanism', type: 'external' },
    { id: 'n12', label: 'Symbolic AI', type: 'external' },
    { id: 'n13', label: 'Cognitive science', type: 'external' },
    { id: 'n14', label: 'Graph databases', type: 'external' },
    { id: 'n15', label: 'LLM research', type: 'external' },
  ];
  const links = [
    { source: 'n1', target: 'n2' }, { source: 'n1', target: 'n12' },
    { source: 'n1', target: 'n3' }, { source: 'n2', target: 'n11' },
    { source: 'n3', target: 'n6' }, { source: 'n3', target: 'n13' },
    { source: 'n4', target: 'n5' }, { source: 'n4', target: 'n9' },
    { source: 'n5', target: 'n10' }, { source: 'n5', target: 'n13' },
    { source: 'n6', target: 'n14' }, { source: 'n7', target: 'n15' },
    { source: 'n7', target: 'n3' }, { source: 'n8', target: 'n4' },
    { source: 'n8', target: 'n1' }, { source: 'n9', target: 'n13' },
    { source: 'n2', target: 'n7' }, { source: 'n6', target: 'n7' },
  ];
  return { nodes, links };
}

function makeDemoPersona() {
  return {
    corpus_size: { note_count: 150 },
    llm_self_description: "This intellectual corpus belongs to someone standing at the crossroads of cognitive science, artificial intelligence, and philosophy of mind. The writing reveals a thinker drawn to foundational questions — not merely how systems work, but why they mean anything at all. There is a recurring preoccupation with memory: biological, computational, distributed. The prose is careful without being cautious, speculative without being reckless. At the edges: an engineer who reads philosophy, or a philosopher who builds things.",
    topical_fingerprint: {
      top_tags: {
        'AI': 42, 'cognition': 35, 'philosophy': 28,
        'memory': 24, 'engineering': 20, 'language': 18,
        'systems': 15, 'epistemics': 12,
      }
    },
    stance_map: {
      'Consciousness': 'Functionalism is probably right, but the hard problem is a genuine explanatory gap, not a category error.',
      'AI Alignment': 'The problem is real and underspecified. Current approaches conflate corrigibility with alignment.',
      'Extended Cognition': 'The extended mind thesis is compelling as a philosophical claim; less clear as a neuroscience claim.',
      'Knowledge Representation': 'Hybrid neuro-symbolic architectures are the right research direction for the next decade.',
    }
  };
}

/* ── Router ────────────────────────────────────────────────── */
function navigate(view) {
  if (!view) return;
  // 1. Reset all views and links
  $$('.view').forEach(el => { el.classList.remove('view--active'); el.style.display = ''; });
  $$('.nav-link').forEach(el => el.classList.remove('active'));

  const viewEl = $(`#view-${view}`);
  if (!viewEl) return;

  // 2. Set State and handle "Brain" special display
  State.activeView = view;
  viewEl.style.display = view === 'brain' ? 'block' : '';
  
  // 3. THE ULTIMATE SCROLL FIX: Immediate snap + 100ms safety net
  window.scrollTo(0, 0); 
  setTimeout(() => window.scrollTo(0, 0), 100);

  // 4. Trigger CSS transition
  void viewEl.offsetWidth;
  viewEl.classList.add('view--active');
  $$(`[data-view="${view}"]`).forEach(el => el.classList.add('active'));

  // 5. Initialize each section
  if (view === 'garden')  initGarden();
  if (view === 'brain')   initBrain();
  if (view === 'profile') initProfile();
  if (view === 'about')   initAbout(); // <--- Keeps your homepage counters working!

  // 6. Update URL
  history.replaceState(null, '', `#${view}`);
}

function initRouting() {
  // Click on any data-view element
  document.addEventListener('click', e => {
    const el = e.target.closest('[data-view]');
    if (el) {
      e.preventDefault();
      navigate(el.dataset.view);
    }
  });

  // Hash change
  window.addEventListener('hashchange', () => {
    const view = location.hash.slice(1) || 'about';
    navigate(view);
  });

  // Initial route
  const initial = location.hash.slice(1) || 'about';
  navigate(initial);
}

/* Nav scroll effect */
function initNavScroll() {
  const nav = $('#site-nav');
  const observer = new IntersectionObserver(
    ([e]) => nav.classList.toggle('scrolled', !e.isIntersecting),
    { rootMargin: `-${(parseInt(getComputedStyle(document.documentElement).getPropertyValue('--nav-h')) || 56) + 10}px 0px 0px 0px` }
  );
  const sentinel = document.createElement('div');
  sentinel.style.cssText = 'position:absolute;top:0;height:1px;width:1px;pointer-events:none';
  document.body.prepend(sentinel);
  observer.observe(sentinel);
}

/* ── About View ────────────────────────────────────────────── */
function initAbout() {
  // Counters — populate once data is loaded
  const animate = () => {
    const notesData = State.notes;
    const personaData = State.persona;
    const graphData  = State.graph;

    const noteCount = notesData?.total
      ?? personaData?.corpus_size?.note_count ?? 0;
    const wordCount = notesData
      ? Object.values(notesData.categories || {}).flat()
          .reduce((s, n) => s + (n.word_count || 0), 0)
      : 0;
    const edgeCount = graphData?.links?.length ?? 0;

    animateCounter($('#cnt-notes'), noteCount);
    animateCounter($('#cnt-words'), wordCount);
    animateCounter($('#cnt-edges'), edgeCount);
  };

  // If data already loaded, animate immediately; else wait
  if (State.notes || State.persona) {
    animate();
  } else {
    // Will be called again after data loads; set placeholders
    setTimeout(animate, 500);
  }
}

/* ── Garden View ───────────────────────────────────────────── */
let gardenInitialized = false;

function initGarden() {
  if (!State.notes) {
    renderGardenLoading();
    return;
  }
  renderGarden();
  if (!gardenInitialized) {
    gardenInitialized = true;
    setupGardenControls();
  }
}

function renderGardenLoading() {
  const grid = $('#garden-grid');
  if (grid) grid.innerHTML = '<div class="state-loading">Loading notes…</div>';
}

function setupGardenControls() {
  // Search
  const searchEl = $('#garden-search');
  if (searchEl) {
    searchEl.addEventListener('input', () => {
      State.gardenQuery = searchEl.value.toLowerCase().trim();
      renderGarden();
    });
  }

  // Category filters
  const filtersEl = $('#cat-filters');
  const cats = Object.keys(State.notes?.categories || {});
  if (filtersEl && cats.length) {
    const allBtn = makeFilterBtn('All', true);
    filtersEl.appendChild(allBtn);
    cats.forEach(cat => filtersEl.appendChild(makeFilterBtn(cat, false)));
  }
}

function makeFilterBtn(label, isActive) {
  const btn = document.createElement('button');
  btn.className = 'cat-btn' + (isActive ? ' active' : '');
  btn.textContent = label;
  btn.addEventListener('click', () => {
    $$('.cat-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    State.gardenCategory = label === 'All' ? 'all' : label;
    renderGarden();
  });
  return btn;
}

function getAllNotes() {
  if (!State.notes?.categories) return [];
  return Object.entries(State.notes.categories).flatMap(([cat, notes]) =>
    notes.map(n => ({ ...n, category: cat }))
  );
}

function renderGarden() {
  const notes = getAllNotes();
  const grid = $('#garden-grid');
  const countEl = $('#garden-count');
  if (!grid) return;

  let filtered = notes;

  // Category filter
  if (State.gardenCategory !== 'all') {
    filtered = filtered.filter(n => n.category === State.gardenCategory);
  }

  // Search filter
  if (State.gardenQuery) {
    filtered = filtered.filter(n =>
      n.title?.toLowerCase().includes(State.gardenQuery) ||
      n.content?.toLowerCase().includes(State.gardenQuery) ||
      n.tags?.some(t => t.toLowerCase().includes(State.gardenQuery))
    );
  }

  if (countEl) {
    countEl.textContent = `${filtered.length} note${filtered.length !== 1 ? 's' : ''}`;
  }

  if (filtered.length === 0) {
    grid.innerHTML = '<div class="garden-empty">No notes found.</div>';
    return;
  }

  grid.innerHTML = '';
  filtered.forEach((note, i) => {
    const card = document.createElement('div');
    card.className = 'note-card';
    card.style.animationDelay = `${Math.min(i * 40, 400)}ms`;
    card.innerHTML = `
      <div class="card-category">${escapeHtml(note.category || '')}</div>
      <div class="card-title">${escapeHtml(note.title || 'Untitled')}</div>
      <div class="card-excerpt">${escapeHtml(note.content?.slice(0, 200) || '')}</div>
      <div class="card-meta">
        ${note.date ? `<span class="card-date">${formatDate(note.date)}</span>` : ''}
        ${note.word_count ? `<span class="card-words">${note.word_count.toLocaleString()} words</span>` : ''}
      </div>
      ${note.tags?.length ? `<div class="card-tags">${note.tags.map(t => `<span class="card-tag">${escapeHtml(t)}</span>`).join('')}</div>` : ''}
    `;
    card.addEventListener('click', () => openNote(note));
    grid.appendChild(card);
  });
}

/* ── Note Reader ───────────────────────────────────────────── */
function openNote(note) {
  const reader = $('#note-reader');
  const titleEl = $('#reader-title');
  const bodyEl  = $('#reader-body');
  const metaEl  = $('#reader-meta');

  if (!reader) return;

  titleEl.textContent = note.title || 'Untitled';
  metaEl.textContent  = [
    note.category,
    formatDate(note.date),
    note.word_count ? `${note.word_count.toLocaleString()} words` : null,
  ].filter(Boolean).join(' · ');

  bodyEl.innerHTML = renderMarkdown(note.content || '');

  reader.hidden = false;
  document.body.style.overflow = 'hidden';

  // Focus for accessibility
  reader.focus?.();
}

function closeNote() {
  const reader = $('#note-reader');
  if (!reader) return;
  reader.hidden = true;
  document.body.style.overflow = '';
}

function initReader() {
  $('#reader-back')?.addEventListener('click', closeNote);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !$('#note-reader').hidden) closeNote();
  });
}

/* ── Brain / D3 Graph ──────────────────────────────────────── */
let brainInitialized = false;

function initBrain() {
  if (brainInitialized) return;
  if (typeof d3 === 'undefined') {
    // D3 deferred — retry when ready
    const check = setInterval(() => {
      if (typeof d3 !== 'undefined') { clearInterval(check); buildGraph(); }
    }, 100);
    return;
  }
  buildGraph();
}

  function buildGraph() {
  if (brainInitialized) return;
  brainInitialized = true;

  const data = State.graph || makeDemoGraph();
  State.graphData = { ...data };

  const nodes = data.nodes.map(n => ({ ...n }));
  
  const clusterMap = {};
  nodes.forEach(n => { clusterMap[n.id] = n.cluster; });
  
  // ULTRA-LIGHT EDGES: Only top 800 connections. 
  // This makes the clusters visible but keeps FPS very high.
  const links = data.links
    .map(l => ({ ...l }))
    .filter(l => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      return clusterMap[s] !== undefined && clusterMap[s] === clusterMap[t];
    })
    .sort((a, b) => (b.weight || 0) - (a.weight || 0))
    .slice(0, 800);

  const svg = d3.select('#graph-svg');
  const width = () => svg.node().clientWidth;
  const height = () => svg.node().clientHeight;

  // Lightweight glow filter — single-pass feGaussianBlur, safe at 2000 nodes
  const defs = svg.append('defs');
  const glowFilter = defs.append('filter')
    .attr('id', 'node-glow')
    .attr('x', '-60%').attr('y', '-60%')
    .attr('width', '220%').attr('height', '220%');
  glowFilter.append('feGaussianBlur')
    .attr('in', 'SourceGraphic').attr('stdDeviation', '2.5').attr('result', 'blur');
  const feMerge = glowFilter.append('feMerge');
  feMerge.append('feMergeNode').attr('in', 'blur');
  feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

  const zoom = d3.zoom()
    .scaleExtent([0.1, 10])
    .on('zoom', ({ transform }) => g.attr('transform', transform));
  svg.call(zoom);
  State.graphZoom = zoom;

  const g = svg.append('g');

  // THE "BALL" INITIAL SHAPE: Random scatter in a central disk
  const cx = width() / 2;
  const cy = height() / 2;
  nodes.forEach((n) => {
    const r = Math.random() * 200;
    const a = Math.random() * Math.PI * 2;
    n.x = cx + r * Math.cos(a);
    n.y = cy + r * Math.sin(a);
    n.fx = n.x; 
    n.fy = n.y; 
  });

  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(50).strength(0.8))
    .force('charge', d3.forceManyBody().strength(-150))
    .force('center', d3.forceCenter(cx, cy))
    .force('collision', d3.forceCollide(12))
    .force("clusterX", d3.forceX(d => {
        if (!State.graphClusterMode) return cx;
        return cx + (Math.cos((d.cluster || 0) * 0.5) * 300); 
    }).strength(d => State.graphClusterMode ? 0.5 : 0.05))
    .force("clusterY", d3.forceY(d => {
        if (!State.graphClusterMode) return cy;
        return cy + (Math.sin((d.cluster || 0) * 0.5) * 300);
    }).strength(d => State.graphClusterMode ? 0.5 : 0.05))
    .alphaDecay(0.05); // Makes it settle faster

  sim.stop(); 
  State.graphSim = sim;

  const linkSel = g.append('g').attr('class', 'links')
    .selectAll('line').data(links).join('line').attr('class', 'link');

  // Normalize centrality so radius is always visible regardless of PageRank scale.
  // PageRank on 2000 nodes gives values ~0.0002–0.005 — multiply by raw r would give <3px.
  const maxCentrality = Math.max(...nodes.map(n => n.centrality || 0), 0.0001);

  const nodeSel = g.append('g').attr('class', 'nodes')
    .selectAll('circle')
    .data(nodes)
    .join('circle')
    // sqrt of normalised centrality → compresses the range, small nodes stay visible
    .attr('r', d => 5 + Math.sqrt((d.centrality || 0) / maxCentrality) * 13)
    .attr('class', d => `node-circle node--${d.role === 'output' ? 'mine' : 'external'}`)
    // Inline fill & stroke — bypasses the CSS selector mismatch
    // (CSS was targeting `.node--mine circle` but the circle IS the element)
    .attr('fill',         d => d.role === 'output' ? '#e8b84b' : '#7b9fd4')
    .attr('stroke',       d => d.role === 'output' ? 'rgba(240,185,70,0.45)' : 'rgba(110,155,220,0.35)')
    .attr('stroke-width', 1.5)
    .attr('filter', 'url(#node-glow)')
    .call(d3.drag()
      .on('start', (event, d) => { if (!event.active && !State.graphFrozen) sim.alphaTarget(0.2).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag',  (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on('end',   (event, d) => { if (!event.active) sim.alphaTarget(0); if (!State.graphFrozen) { d.fx = null; d.fy = null; } })
    );

  // TEXT LABELS REMOVED FOR PERFORMANCE

  sim.on('tick', () => {
    linkSel.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
           .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    nodeSel.attr('cx', d => d.x).attr('cy', d => d.y);
  });

  const tooltip = $('#graph-tooltip');
  nodeSel
    .on('mouseenter', (event, d) => {
      if (tooltip) {
        tooltip.innerHTML = `
          <div class="tooltip-label">${escapeHtml(d.title || "Untitled Note")}</div>
          <div class="tooltip-type">${d.role === 'output' ? '◉ Your note' : '○ Reference'}</div>
          ${d.snippet ? `<div class="tooltip-excerpt">${escapeHtml(d.snippet)}</div>` : ''}
        `;
        tooltip.classList.add('visible');
      }
    })
    .on('mouseleave', () => { if (tooltip) tooltip.classList.remove('visible'); });

  // UI CONTROLS
  // Meta count
  const metaEl = $('#graph-meta');
  if (metaEl) metaEl.textContent = `${nodes.length} nodes · ${links.length} edges`;

  // Fit
  $('#btn-zoom-fit')?.addEventListener('click', () => fitGraph(svg, g, zoom));

  // Freeze / Play Physics
  const btnFreeze = $('#btn-freeze');
  if (btnFreeze) {
    btnFreeze.classList.add('active');
    btnFreeze.innerHTML = '▶ Play Physics';
    btnFreeze.addEventListener('click', function() {
      State.graphFrozen = !State.graphFrozen;
      this.classList.toggle('active', State.graphFrozen);
      if (State.graphFrozen) {
        this.innerHTML = '▶ Play Physics';
        sim.stop();
        nodes.forEach(n => { n.fx = n.x; n.fy = n.y; });
      } else {
        this.innerHTML = '⏸ Freeze';
        nodes.forEach(n => { n.fx = null; n.fy = null; });
        sim.alpha(1).restart();
      }
    });
  }

  // Mine only — handles both real data (role:'output') and demo data (type:'mine')
  $('#btn-mine')?.addEventListener('click', function() {
    State.graphMineOnly = !State.graphMineOnly;
    this.classList.toggle('active', State.graphMineOnly);
    const isMine = d => d.role === 'output' || d.type === 'mine';
    if (State.graphMineOnly) {
      nodeSel.classed('node--dimmed', d => !isMine(d));
      linkSel.style('opacity', l => {
        const s = typeof l.source === 'object' ? l.source : nodes.find(n => n.id === l.source);
        const t = typeof l.target === 'object' ? l.target : nodes.find(n => n.id === l.target);
        return isMine(s) && isMine(t) ? 1 : 0.04;
      });
    } else {
      nodeSel.classed('node--dimmed', false);
      linkSel.style('opacity', null);
    }
  });

  // Cluster Map Toggle
  $('#btn-cluster')?.addEventListener('click', function() {
    State.graphClusterMode = !State.graphClusterMode;
    this.classList.toggle('active', State.graphClusterMode);
    
    if (State.graphClusterMode) {
      this.style.color = '#c8a86b'; // Your cosmic gold
      this.style.borderColor = 'rgba(200,168,107,0.4)';
    } else {
      this.style.color = '';
      this.style.borderColor = '';
    }
    
    // Automatically unfreeze the graph so the animation plays
    if (State.graphFrozen) {
      State.graphFrozen = false;
      const btnF = $('#btn-freeze');
      if (btnF) {
        btnF.classList.remove('active');
        btnF.innerHTML = '⏸ Freeze';
      }
    }
    
    // Unpin nodes and wake up physics
    nodes.forEach(n => { n.fx = null; n.fy = null; });
    sim.alpha(0.8).restart();
  });

  // Graph search — checks title (real data) and label (demo data)
  $('#graph-search')?.addEventListener('input', function() {
    const q = this.value.toLowerCase().trim();
    if (!q) { nodeSel.classed('node--dimmed', false); return; }
    nodeSel.classed('node--dimmed', d =>
      !(d.title || d.label || d.id || '').toLowerCase().includes(q)
    );
  });

  // Resize: re-centre force on viewport change
  new ResizeObserver(() => {
    sim.force('center', d3.forceCenter(width() / 2, height() / 2));
  }).observe(svg.node());

  setTimeout(() => fitGraph(svg, g, zoom), 300);
}


function fitGraph(svg, g, zoom) {
  try {
    const bounds = g.node().getBBox();
    if (!bounds.width || !bounds.height) return;
    const w = svg.node().clientWidth;
    const h = svg.node().clientHeight;
    const padding = 60;
    const scale = Math.min(
      (w - padding * 2) / bounds.width,
      (h - padding * 2) / bounds.height,
      2
    );
    const tx = w / 2 - scale * (bounds.x + bounds.width / 2);
    const ty = h / 2 - scale * (bounds.y + bounds.height / 2);
    svg.transition().duration(600).call(
      zoom.transform,
      d3.zoomIdentity.translate(tx, ty).scale(scale)
    );
  } catch {}
}

/* ── Profile View ──────────────────────────────────────────── */
/* ── Profile View ──────────────────────────────────────────── */
/* ── Profile View ──────────────────────────────────────────── */

let profileInitialized = false;
function initProfile() {
  if (profileInitialized || !State.persona) return;
  profileInitialized = true;

  const p = State.persona;

  // Date & Description
  const dateEl = $('#profile-date');
  if (dateEl) dateEl.textContent = `Generated ${formatDate(new Date().toISOString())}`;
  const descEl = $('#profile-desc');
  if (descEl && p.llm_self_description) descEl.textContent = p.llm_self_description;

  // Topical fingerprint
  const topicsEl = $('#topics-chart');
  if (topicsEl && p.topical_fingerprint?.top_tags) {
    const tags = Object.entries(p.topical_fingerprint.top_tags)
      .sort((a, b) => b[1] - a[1]).slice(0, 8);
    const max = tags[0]?.[1] || 1;
    topicsEl.innerHTML = tags.map(([tag, count]) => `
      <div class="topic-row">
        <div class="topic-label-row"><span class="topic-name">${escapeHtml(tag)}</span><span class="topic-count">${count}</span></div>
        <div class="topic-bar-bg"><div class="topic-bar-fill" data-pct="${(count / max * 100).toFixed(1)}"></div></div>
      </div>
    `).join('');
    requestAnimationFrame(() => { $$('.topic-bar-fill').forEach(bar => { bar.style.width = bar.dataset.pct + '%'; }); });
  }

  // Stance map
  const stancesEl = $('#stances-list');
  if (stancesEl && p.stance_map) {
    stancesEl.innerHTML = Object.entries(p.stance_map).map(([topic, stance]) => `
      <div class="stance-item">
        <div class="stance-topic">${escapeHtml(topic)}</div>
        <div class="stance-text">${escapeHtml(stance)}</div>
      </div>
    `).join('');

    // --- STYLISTIC DNA (WITH DEEP DIVE TOGGLE) ---
    if (p.stylistic_markers && p.argument_patterns) {
      const styleContainer = document.createElement('div');
      styleContainer.className = 'profile-section';
      styleContainer.style.marginTop = '40px';
      
      const args = Object.entries(p.argument_patterns).sort((a, b) => b[1] - a[1]);
      const topArgs = args.slice(0, 3).map(a => a[0].replace(/_/g, ' '));
      
      // Format Intellectual Lineage
      const lineage = p.intellectual_lineage?.cited_figures || {};
      const topCited = Object.entries(lineage).slice(0, 6).map(x => x[0]).join('<span style="color:#6b7280;"> · </span>') || 'No citations detected';

      // Format Punctuation
      const punct = p.stylistic_markers.punctuation_style || {};
      const punctList = Object.entries(punct).map(([k,v]) => `${k.replace('uses_', '').replace('_', ' ')}: <span style="color:#fff">${v}</span>`).join('<br>');

      styleContainer.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; margin-bottom: 20px;">
          <h3 style="color: #c8a86b; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin: 0;">Stylistic & Argument DNA</h3>
          <button id="btn-deep-dna" style="background: none; border: 1px solid rgba(255,255,255,0.2); color: #a0aabf; font-size: 11px; padding: 4px 10px; border-radius: 4px; cursor: pointer; text-transform: uppercase;">Expand Deep Data ↴</button>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; color: #a0aabf; font-size: 14px;">
          <div>
            <strong>Rhythm & Vocabulary</strong><br>
            <span style="color: #fff;">Sentence length:</span> ${p.stylistic_markers.avg_sentence_length || '?'} words avg<br>
            <span style="color: #fff;">Word length:</span> ${p.stylistic_markers.avg_word_length || '?'} chars avg<br>
            <span style="color: #fff;">Lexical richness:</span> ${((p.stylistic_markers.vocabulary_richness || 0) * 100).toFixed(1)}% unique
          </div>
          <div>
            <strong>Dominant Logical Structures</strong><br>
            <span style="color: #fff;">1.</span> ${topArgs[0] || 'N/A'}<br>
            <span style="color: #fff;">2.</span> ${topArgs[1] || 'N/A'}<br>
            <span style="color: #fff;">3.</span> ${topArgs[2] || 'N/A'}
          </div>
        </div>

        <div id="deep-dna-panel" style="display: none; margin-top: 25px; padding-top: 20px; border-top: 1px dashed rgba(255,255,255,0.15); font-size: 13px; color: #a0aabf;">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            
            <div>
              <strong style="color: #6577a8; display: block; margin-bottom: 6px;">Intellectual Lineage</strong>
              <div style="margin-bottom: 15px; color: #e0e0e0;">${topCited}</div>
              
              <strong style="color: #6577a8; display: block; margin-bottom: 6px;">Punctuation Habits</strong>
              <div style="color: #a0aabf;">${punctList || 'Not enough data'}</div>
            </div>

            <div>
              <strong style="color: #6577a8; display: block; margin-bottom: 6px;">Complete Logical Topology</strong>
              ${args.map(([name, val]) => `
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                  <span>${name.replace(/_/g, ' ')}</span>
                  <span style="color: #fff;">${(val * 100).toFixed(1)}%</span>
                </div>
              `).join('')}
            </div>

          </div>
        </div>
      `;
      stancesEl.parentNode.appendChild(styleContainer);

      // Add toggle logic
      document.getElementById('btn-deep-dna').addEventListener('click', function() {
        const panel = document.getElementById('deep-dna-panel');
        if (panel.style.display === 'none') {
          panel.style.display = 'block';
          this.innerHTML = 'Hide Deep Data ↰';
          this.style.color = '#c8a86b';
          this.style.borderColor = 'rgba(200,168,107,0.4)';
        } else {
          panel.style.display = 'none';
          this.innerHTML = 'Expand Deep Data ↴';
          this.style.color = '#a0aabf';
          this.style.borderColor = 'rgba(255,255,255,0.2)';
        }
      });
    }
  }
}
/* ── Bootstrap ─────────────────────────────────────────────── */
async function bootstrap() {
  // Fetch all data in parallel
  const [notesRaw, graphRaw, personaRaw] = await Promise.all([
    fetchJSON('notes.json'),
    fetchJSON('graph_data.json'),
    fetchJSON('persona.json'),
  ]);

  // Use real data or fall back to demos
  State.notes   = notesRaw   || makeDemoNotes();
  State.graph   = graphRaw   || makeDemoGraph();
  State.persona = personaRaw || makeDemoPersona();

  // If active view is waiting for data, re-init
  const v = State.activeView;
  if (v === 'about')   initAbout();
  if (v === 'garden')  { gardenInitialized = false; initGarden(); setupGardenControls(); }
  if (v === 'brain')   { brainInitialized = false; initBrain(); }
  if (v === 'profile') { profileInitialized = false; initProfile(); }
}

/* ── Init ──────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initNavScroll();
  initRouting();
  initReader();
  bootstrap();
});
