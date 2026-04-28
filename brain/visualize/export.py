"""
Graph Exporter
--------------
Generates the JSON payload required by D3.js to render the interactive web graph.
Can be run locally or used to push updates to your WordPress site.
"""

import json
import logging
from pathlib import Path

from ..memory.store import Store
from ..memory.graph import GraphBuilder

log = logging.getLogger(__name__)

class GraphExporter:
    def __init__(self, store: Store, builder: GraphBuilder):
        self.store = store
        self.builder = builder

    def export_json(self, output_path: str = "web/graph_data.json"):
        """Builds the graph and exports the nodes/links to a JSON file."""
        log.info("[export] Generating graph JSON for visualization...")
        
        # Build the full graph (explicit links + tags + semantic)
        G = self.builder.build(use_explicit=True, use_tags=True, use_semantic=True)
        
        # We need clusters and centrality to color and size the nodes visually
        self.builder.compute_clusters(G)
        self.builder.compute_centrality(G)
        
        # Convert to D3.js compatible dictionary
        data = self.builder.to_json(G)
        
        # Ensure output directory exists
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to disk
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        log.info(f"[export] Successfully saved {len(data['nodes'])} nodes and {len(data['links'])} links to {output_path}")
        return data
