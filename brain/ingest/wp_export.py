"""
WordPress Exporter
------------------
Publishes your Digital Brain to WordPress via the REST API.

Two publish modes:

  graph   — Exports the D3 force graph as a self-contained HTML embed
            and creates/updates a single WordPress page with it.
            The graph is an <iframe> pointing to a CDN-hosted or inline
            version of graph_data.json.

  wiki    — Publishes each auto-wiki page as a WordPress post
            (or updates it if it already exists, matched by slug).
            Posts are tagged with the concept name and placed in a
            configurable category.

Authentication:
  Uses WordPress Application Passwords (WP 5.6+).
  Go to: Users → Your Profile → Application Passwords → Add New.
  The credential is: "username:application_password" base64-encoded.

Usage (CLI via main.py):
  python main.py export-wp                    # publish graph page
  python main.py export-wp --mode wiki        # publish all wiki pages as posts
  python main.py export-wp --mode both        # both
  python main.py export-wp --dry-run          # print what would be published

Usage (programmatic):
  exporter = WPExporter(store, builder, wiki, cfg)
  exporter.publish_graph()
  exporter.publish_wiki_pages(category="Digital Brain")
"""

import base64
import json
import logging
import os
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from brain.memory.store import Store
from brain.memory.graph import GraphBuilder
from brain.visualize.export import GraphExporter

log = logging.getLogger(__name__)


# ─── HTML template for the embedded graph ────────────────────────────────────

_GRAPH_EMBED_TEMPLATE = """\
<!-- Digital Brain Graph — auto-generated {date} -->
<div id="digital-brain-graph" style="width:100%;height:600px;background:#111;border-radius:8px;overflow:hidden;">
<iframe
  src="{graph_url}"
  width="100%"
  height="600"
  frameborder="0"
  style="border-radius:8px;"
  title="Digital Brain Knowledge Graph"
  loading="lazy"
></iframe>
</div>
<p style="font-size:0.8rem;color:#888;text-align:right;">
  {n_nodes} concepts · {n_edges} connections ·
  <a href="{graph_url}" target="_blank">open full screen ↗</a>
</p>
"""

# ─── Inline self-contained graph (no iframe) ────────────────────────────────

_INLINE_GRAPH_TEMPLATE = """\
<!-- Digital Brain — inline graph embed — auto-generated {date} -->
<div id="db-graph-{uid}" style="width:100%;height:580px;background:#111;border-radius:8px;position:relative;">
<script>
(function(){{
  var DATA = {graph_json};
  // Minimal D3 force graph — requires d3.v7 to be loaded on the page
  // If D3 is not available, this falls back to a plain list.
  if(typeof d3 === 'undefined') {{
    var el = document.getElementById('db-graph-{uid}');
    el.style.background = '#1a1a2e';
    el.style.padding = '20px';
    el.style.color = '#eee';
    var html = '<h3 style="margin-top:0">Knowledge Graph ({n_nodes} concepts)</h3><ul>';
    DATA.nodes.slice(0,30).forEach(function(n){{
      html += '<li>' + n.title + '</li>';
    }});
    html += '</ul>';
    el.innerHTML = html;
    return;
  }}
  var container = document.getElementById('db-graph-{uid}');
  var w = container.offsetWidth, h = 580;
  var svg = d3.select(container).append('svg').attr('width',w).attr('height',h)
    .call(d3.zoom().on('zoom', function(e){{ g.attr('transform', e.transform); }}));
  var g = svg.append('g');
  var color = d3.scaleOrdinal(d3.schemeCategory10);
  var sim = d3.forceSimulation(DATA.nodes)
    .force('link', d3.forceLink(DATA.links).id(function(d){{return d.id;}}).distance(45))
    .force('charge', d3.forceManyBody().strength(-80))
    .force('center', d3.forceCenter(w/2, h/2))
    .force('collide', d3.forceCollide().radius(12));
  var link = g.append('g').selectAll('line').data(DATA.links).join('line')
    .attr('stroke', function(d){{ return d.edge_type==='llm'?'#ff5555':'#444'; }})
    .attr('stroke-width', function(d){{ return Math.max(0.5, d.weight*2); }})
    .attr('stroke-opacity', 0.6);
  var node = g.append('g').selectAll('circle').data(DATA.nodes).join('circle')
    .attr('r', function(d){{ return Math.max(4, Math.sqrt((d.centrality||0.01)*2000)); }})
    .attr('fill', function(d){{ return color(d.cluster||0); }})
    .attr('stroke','#fff').attr('stroke-width',1)
    .call(d3.drag()
      .on('start',function(e,d){{ if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
      .on('drag', function(e,d){{ d.fx=e.x; d.fy=e.y; }})
      .on('end',  function(e,d){{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }}));
  node.append('title').text(function(d){{ return d.title; }});
  sim.on('tick', function(){{
    link.attr('x1',function(d){{return d.source.x;}}).attr('y1',function(d){{return d.source.y;}})
        .attr('x2',function(d){{return d.target.x;}}).attr('y2',function(d){{return d.target.y;}});
    node.attr('cx',function(d){{return d.x;}}).attr('cy',function(d){{return d.y;}});
  }});
}})();
</script>
</div>
<script src="https://d3js.org/d3.v7.min.js" defer></script>
"""


class WPExporter:
    """
    Publishes Digital Brain content to WordPress via REST API.

    Parameters
    ----------
    store   : Store
    builder : GraphBuilder
    wiki    : AutoWiki (can be None if not publishing wiki pages)
    cfg     : dict — must include wp_url, wp_user, wp_app_password
    """

    def __init__(self, store: Store, builder: GraphBuilder,
                 wiki=None, cfg: dict = None):
        self.store   = store
        self.builder = builder
        self.wiki    = wiki
        self.cfg     = cfg or {}

        self.wp_url  = self.cfg.get("wp_url", os.environ.get("WP_URL", "")).rstrip("/")
        self.wp_user = self.cfg.get("wp_user", os.environ.get("WP_USER", ""))
        self.wp_pass = self.cfg.get("wp_app_password",
                                    os.environ.get("WP_APP_PASSWORD", ""))

        if not all([self.wp_url, self.wp_user, self.wp_pass]):
            log.warning("[wp] WP_URL, WP_USER, and WP_APP_PASSWORD must be set. "
                        "See: Users → Profile → Application Passwords in WordPress.")

    # ── Public API ────────────────────────────────────────────────────────────

    def publish_graph(self, page_title: str = "Knowledge Graph",
                      page_slug: str = "knowledge-graph",
                      embed_mode: str = "inline",
                      graph_url: str = "",
                      dry_run: bool = False) -> Optional[dict]:
        """
        Export graph JSON and publish/update a WordPress page.

        embed_mode:
          'inline' — embeds the graph directly in the page (no iframe)
          'iframe'  — links to graph_url (you must host graph_data.json separately)
        """
        log.info("[wp] Building graph for export...")
        graph_exporter = GraphExporter(self.store, self.builder)
        data = graph_exporter.export_json("web/graph_data.json")

        n_nodes = len(data.get("nodes", []))
        n_edges = len(data.get("links", []))
        date    = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if embed_mode == "inline":
            import uuid
            uid       = uuid.uuid4().hex[:8]
            graph_json = json.dumps(data)
            content = _INLINE_GRAPH_TEMPLATE.format(
                date=date, uid=uid, graph_json=graph_json,
                n_nodes=n_nodes,
            )
        else:
            if not graph_url:
                log.error("[wp] iframe mode requires --graph-url pointing to hosted graph_data.json")
                return None
            content = _GRAPH_EMBED_TEMPLATE.format(
                date=date, graph_url=graph_url,
                n_nodes=n_nodes, n_edges=n_edges,
            )

        if dry_run:
            print(f"[DRY RUN] Would publish page: '{page_title}' ({n_nodes} nodes, {n_edges} edges)")
            print(f"  URL: {self.wp_url}/wp-json/wp/v2/pages")
            return {"dry_run": True, "title": page_title, "n_nodes": n_nodes}

        return self._upsert_page(
            title=page_title,
            slug=page_slug,
            content=content,
            meta={"description": f"Auto-generated knowledge graph. {n_nodes} concepts, {n_edges} connections."},
        )

    def publish_wiki_pages(self, category: str = "Digital Brain",
                           status: str = "publish",
                           dry_run: bool = False) -> list:
        """
        Publish each auto-wiki Note as a WordPress post.
        Updates existing posts if a matching slug is found.
        """
        if not self.wiki:
            log.error("[wp] No AutoWiki instance provided.")
            return []

        pages  = self.wiki.list_pages()
        cat_id = self._get_or_create_category(category) if not dry_run else 0
        results = []

        for page in pages:
            concept = page.metadata.get("wiki_concept", page.title)
            slug    = f"brain-wiki-{re.sub(r'[^\\w-]', '-', concept.lower())}"
            version = page.metadata.get("version", 1)

            # Convert [[wikilinks]] to HTML links
            content_html = self._wikilinks_to_html(page.content)
            # Convert markdown-ish to HTML
            content_html = self._md_to_html(content_html)

            excerpt = page.short_content(200) if hasattr(page, "short_content") else page.content[:200]

            if dry_run:
                print(f"[DRY RUN] Would publish post: '{page.title}' (v{version})")
                results.append({"dry_run": True, "title": page.title, "slug": slug})
                continue

            result = self._upsert_post(
                title=page.title,
                slug=slug,
                content=content_html,
                excerpt=excerpt,
                category_ids=[cat_id] if cat_id else [],
                tags=[concept, "wiki", "digital-brain"],
                status=status,
            )
            if result:
                log.info(f"[wp] Published: {page.title} → {result.get('link', '')}")
                results.append(result)

        log.info(f"[wp] Published {len(results)} wiki pages.")
        return results

    # ── WordPress REST API ────────────────────────────────────────────────────

    def _auth_header(self) -> str:
        cred = f"{self.wp_user}:{self.wp_pass}"
        return "Basic " + base64.b64encode(cred.encode()).decode()

    def _request(self, method: str, endpoint: str, body: dict = None) -> Optional[dict]:
        url     = f"{self.wp_url}/wp-json/wp/v2/{endpoint}"
        payload = json.dumps(body).encode("utf-8") if body else None
        req     = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": self._auth_header(),
                "Content-Type":  "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")[:300]
            log.error(f"[wp] HTTP {e.code} {method} {endpoint}: {body_text}")
            return None
        except Exception as e:
            log.error(f"[wp] Request failed {method} {endpoint}: {e}")
            return None

    def _find_by_slug(self, post_type: str, slug: str) -> Optional[dict]:
        """Find an existing post/page by slug."""
        results = self._request("GET", f"{post_type}?slug={slug}&per_page=1")
        if results and isinstance(results, list) and results:
            return results[0]
        return None

    def _upsert_page(self, title: str, slug: str, content: str,
                     meta: dict = None) -> Optional[dict]:
        existing = self._find_by_slug("pages", slug)
        body = {
            "title":   title,
            "slug":    slug,
            "content": content,
            "status":  "publish",
            "meta":    meta or {},
        }
        if existing:
            log.info(f"[wp] Updating existing page (id={existing['id']}): {title}")
            return self._request("POST", f"pages/{existing['id']}", body)
        else:
            log.info(f"[wp] Creating new page: {title}")
            return self._request("POST", "pages", body)

    def _upsert_post(self, title: str, slug: str, content: str,
                     excerpt: str = "", category_ids: list = None,
                     tags: list = None, status: str = "publish") -> Optional[dict]:
        # Resolve tag IDs
        tag_ids = []
        for tag_name in (tags or []):
            tag_obj = self._get_or_create_tag(tag_name)
            if tag_obj:
                tag_ids.append(tag_obj)

        body = {
            "title":      title,
            "slug":       slug,
            "content":    content,
            "excerpt":    excerpt,
            "status":     status,
            "categories": category_ids or [],
            "tags":       tag_ids,
        }
        existing = self._find_by_slug("posts", slug)
        if existing:
            return self._request("POST", f"posts/{existing['id']}", body)
        return self._request("POST", "posts", body)

    def _get_or_create_category(self, name: str) -> Optional[int]:
        results = self._request("GET", f"categories?search={name}&per_page=1")
        if results and isinstance(results, list) and results:
            return results[0]["id"]
        result = self._request("POST", "categories", {"name": name})
        return result["id"] if result else None

    def _get_or_create_tag(self, name: str) -> Optional[int]:
        slug    = re.sub(r"[^\\w-]", "-", name.lower())
        results = self._request("GET", f"tags?search={name}&per_page=1")
        if results and isinstance(results, list) and results:
            return results[0]["id"]
        result = self._request("POST", "tags", {"name": name, "slug": slug})
        return result["id"] if result else None

    # ── Content conversion ────────────────────────────────────────────────────

    def _wikilinks_to_html(self, text: str) -> str:
        """Convert [[concept]] links to WP search links."""
        base = self.wp_url
        def replace(m):
            concept = m.group(1)
            slug    = re.sub(r"[^\\w-]", "-", concept.lower())
            return f'<a href="{base}/?s={urllib.parse.quote(concept)}">{concept}</a>'
        import urllib.parse
        return re.sub(r"\[\[([^\]]+)\]\]", replace, text)

    @staticmethod
    def _md_to_html(text: str) -> str:
        """Minimal Markdown → HTML for wiki pages."""
        lines   = text.split("\n")
        html    = []
        in_list = False
        for line in lines:
            if line.startswith("# "):
                if in_list: html.append("</ul>"); in_list = False
                html.append(f"<h2>{line[2:].strip()}</h2>")
            elif line.startswith("## "):
                if in_list: html.append("</ul>"); in_list = False
                html.append(f"<h3>{line[3:].strip()}</h3>")
            elif line.startswith("### "):
                if in_list: html.append("</ul>"); in_list = False
                html.append(f"<h4>{line[4:].strip()}</h4>")
            elif line.startswith("- "):
                if not in_list: html.append("<ul>"); in_list = True
                html.append(f"<li>{line[2:].strip()}</li>")
            elif line.startswith("**") and line.endswith("**"):
                if in_list: html.append("</ul>"); in_list = False
                html.append(f"<strong>{line.strip('*')}</strong>")
            elif line.strip() == "---":
                if in_list: html.append("</ul>"); in_list = False
                html.append("<hr>")
            elif line.strip() == "":
                if in_list: html.append("</ul>"); in_list = False
                html.append("")
            else:
                if in_list: html.append("</ul>"); in_list = False
                # Inline bold/italic
                l = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
                l = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         l)
                l = re.sub(r"`(.+?)`",        r"<code>\1</code>",     l)
                html.append(f"<p>{l}</p>" if l.strip() else "")

        if in_list:
            html.append("</ul>")
        return "\n".join(html)
