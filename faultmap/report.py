"""Generates interactive HTML reports from FaultMap analysis results.

Reads GML graph files, node ranking CSVs, and JSON dictionaries produced
by the noderank and graphreduce stages, then produces a self-contained
HTML file with an embedded D3.js force-directed causal network visualization,
sortable ranking tables, and edge-weight filtering controls.
"""

import csv
import json
import logging
import os
from pathlib import Path

import networkx as nx

from faultmap import config_setup
from faultmap.type_definitions import RunModes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


class ReportData:
    """Collects all artifacts produced by the noderank / graphreduce stages
    that are required to build an interactive report.
    """

    def __init__(self, mode: RunModes, case: str) -> None:
        self.saveloc, self.caseconfigloc, self.casedir, _ = config_setup.run_setup(
            mode, case
        )
        self.case = case
        self.mode = mode

    # ------------------------------------------------------------------
    def collect(self) -> dict:
        """Walk the noderank output tree and collect every scenario's
        graph + ranking data into a JSON-serialisable dictionary.

        Returns a dict of the form::

            {
                "case": "<case name>",
                "scenarios": {
                    "<scenario>": {
                        "<method>/<sigtype>/<embed>/<typename>/<box>/<dummies>": {
                            "graph": { "nodes": [...], "edges": [...] },
                            "rankings": [ [name, score], ... ],
                        }
                    }
                }
            }
        """
        scenarios_dir = os.path.join(self.saveloc, "noderank", self.case)
        if not os.path.isdir(scenarios_dir):
            logger.warning("No noderank results found at %s", scenarios_dir)
            return {"case": self.case, "scenarios": {}}

        result: dict = {"case": self.case, "scenarios": {}}

        for scenario in sorted(os.listdir(scenarios_dir)):
            scenario_path = os.path.join(scenarios_dir, scenario)
            if not os.path.isdir(scenario_path):
                continue
            entries = _collect_scenario(scenario_path)
            if entries:
                result["scenarios"][scenario] = entries

        return result


def _collect_scenario(scenario_path: str) -> dict:
    """Recursively collect graph + ranking pairs under *scenario_path*."""
    entries: dict = {}

    for dirpath, _dirnames, filenames in os.walk(scenario_path):
        gml_files = [f for f in filenames if f.endswith(".gml")]
        csv_files = [
            f for f in filenames
            if f.startswith("rankinglist_") and f.endswith(".csv")
        ]

        for gml_file in gml_files:
            # Determine which rank method this belongs to
            # graph_<method>.gml  or  <name>_simplified.gml
            stem = gml_file.removesuffix(".gml")
            if stem.endswith("_simplified"):
                graph_variant = "simplified"
                rank_method = stem.removesuffix("_simplified").removeprefix("graph_")
            elif stem.endswith("_lowedge"):
                graph_variant = "lowedge"
                rank_method = stem.removesuffix("_lowedge").removeprefix("graph_")
            else:
                graph_variant = "full"
                rank_method = stem.removeprefix("graph_")

            graph_data = _read_graph(os.path.join(dirpath, gml_file))
            if graph_data is None:
                continue

            # Try to find matching ranking CSV
            ranking_csv = f"rankinglist_{rank_method}.csv"
            rankings: list[list] = []
            if ranking_csv in csv_files:
                rankings = _read_ranking_csv(os.path.join(dirpath, ranking_csv))

            # Build a human-readable key from the relative path
            rel = os.path.relpath(dirpath, scenario_path)
            suffix = f"{graph_variant}_{rank_method}"
            key = f"{rel}/{suffix}" if rel != "." else suffix

            entries[key] = {
                "graph": graph_data,
                "rankings": rankings,
            }

    return entries


def _read_graph(path: str) -> dict | None:
    """Read a GML file and return a JSON-friendly dict of nodes and edges."""
    try:
        g = nx.readwrite.read_gml(path)
    except Exception:
        logger.warning("Could not read GML file: %s", path)
        return None

    nodes = []
    for node_id in g.nodes():
        attrs = dict(g.nodes[node_id])
        nodes.append({"id": str(node_id), **attrs})

    edges = []
    for u, v, data in g.edges(data=True):
        edge: dict = {"source": str(u), "target": str(v)}
        edge.update({k: v for k, v in data.items()})
        edges.append(edge)

    return {"nodes": nodes, "edges": edges}


def _read_ranking_csv(path: str) -> list[list]:
    """Read a ranking CSV (name, score) produced by *writecsv_looprank*."""
    rows: list[list] = []
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        rows.append([row[0], float(row[1])])
                    except ValueError:
                        rows.append(list(row))
    except Exception:
        logger.warning("Could not read ranking CSV: %s", path)
    return rows


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FaultMap Report &ndash; {case}</title>
<style>
:root {{
  --bg: #0d1117; --fg: #c9d1d9; --accent: #58a6ff;
  --card: #161b22; --border: #30363d; --green: #3fb950;
  --orange: #d29922; --red: #f85149;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--fg); padding: 1rem; }}
h1 {{ font-size: 1.4rem; margin-bottom: .6rem; color: var(--accent); }}
h2 {{ font-size: 1.1rem; margin: .8rem 0 .4rem; }}

/* layout */
.controls {{ display: flex; flex-wrap: wrap; gap: .5rem; align-items: center;
             padding: .6rem; background: var(--card); border-radius: 6px;
             border: 1px solid var(--border); margin-bottom: .8rem; }}
.controls label {{ font-size: .82rem; }}
.controls select, .controls input {{ font-size: .82rem; padding: 2px 6px;
  background: var(--bg); color: var(--fg); border: 1px solid var(--border);
  border-radius: 4px; }}
.panels {{ display: grid; grid-template-columns: 1fr 340px; gap: .8rem; }}
@media (max-width: 900px) {{ .panels {{ grid-template-columns: 1fr; }} }}

/* graph canvas */
#graph-panel {{ background: var(--card); border-radius: 6px;
  border: 1px solid var(--border); position: relative; min-height: 500px; }}
svg {{ width: 100%; height: 100%; display: block; }}

/* sidebar */
#sidebar {{ display: flex; flex-direction: column; gap: .6rem; }}
.ranking-card {{ background: var(--card); border-radius: 6px;
  border: 1px solid var(--border); padding: .6rem; overflow: auto; max-height: 500px; }}
table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
th, td {{ text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--border); }}
th {{ cursor: pointer; color: var(--accent); user-select: none; }}
th:hover {{ text-decoration: underline; }}
.bar {{ height: 6px; border-radius: 3px; background: var(--accent); }}

/* tooltip */
.tooltip {{ position: absolute; background: var(--card); border: 1px solid var(--border);
  border-radius: 4px; padding: .4rem .6rem; font-size: .78rem; pointer-events: none;
  opacity: 0; transition: opacity .15s; z-index: 10; }}

/* node labels */
.node-label {{ font-size: 10px; fill: var(--fg); pointer-events: none;
  text-anchor: middle; dominant-baseline: central; }}

/* legend */
.legend {{ font-size: .78rem; padding: .4rem .6rem; background: var(--card);
  border-radius: 6px; border: 1px solid var(--border); }}
.legend-row {{ display: flex; align-items: center; gap: .4rem; margin: 2px 0; }}
.legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; }}

footer {{ margin-top: 1.2rem; font-size: .75rem; color: #484f58; text-align: center; }}
</style>
</head>
<body>
<h1>FaultMap Interactive Report &ndash; {case}</h1>

<div class="controls" id="controls">
  <label>Scenario
    <select id="sel-scenario"></select>
  </label>
  <label>Result set
    <select id="sel-result"></select>
  </label>
  <label>Min edge weight
    <input id="weight-thresh" type="range" min="0" max="1" step="0.01" value="0">
    <span id="weight-val">0.00</span>
  </label>
</div>

<div class="panels">
  <div id="graph-panel">
    <div class="tooltip" id="tooltip"></div>
  </div>
  <div id="sidebar">
    <div class="ranking-card" id="ranking-card">
      <h2>Node Rankings</h2>
      <table id="rank-table"><thead><tr><th data-col="rank">#</th><th data-col="name">Node</th><th data-col="score">Score</th><th data-col="bar"></th></tr></thead><tbody></tbody></table>
    </div>
    <div class="legend" id="legend-box">
      <strong>Legend</strong>
      <div class="legend-row"><div class="legend-swatch" style="background:var(--accent)"></div> Node (size = importance)</div>
      <div class="legend-row"><div class="legend-swatch" style="background:var(--green)"></div> Edge (width = weight)</div>
    </div>
  </div>
</div>

<footer>Generated by FaultMap &middot; <a href="https://doi.org/10.5281/zenodo.2543739" style="color:var(--accent)">Zenodo</a></footer>

<!-- D3.js v7 (minified, MIT licence) loaded from CDN -->
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
// ---- embedded data ----
const DATA = {data_json};

// ---- state ----
let currentScenario = null;
let currentResult = null;
let sortCol = "score";
let sortAsc = false;

// ---- selectors ----
const selScenario = d3.select("#sel-scenario");
const selResult   = d3.select("#sel-result");
const slider      = d3.select("#weight-thresh");
const sliderVal   = d3.select("#weight-val");
const tooltip     = d3.select("#tooltip");

// ---- populate scenario selector ----
const scenarios = Object.keys(DATA.scenarios);
selScenario.selectAll("option").data(scenarios).join("option")
  .attr("value", d => d).text(d => d);
if (scenarios.length) {{ currentScenario = scenarios[0]; populateResults(); }}

selScenario.on("change", function() {{
  currentScenario = this.value; populateResults();
}});
selResult.on("change", function() {{
  currentResult = this.value; render();
}});
slider.on("input", function() {{
  sliderVal.text(parseFloat(this.value).toFixed(2)); render();
}});

function populateResults() {{
  const keys = currentScenario ? Object.keys(DATA.scenarios[currentScenario]) : [];
  selResult.selectAll("option").data(keys).join("option")
    .attr("value", d => d).text(d => d);
  currentResult = keys.length ? keys[0] : null;
  render();
}}

// ---- graph rendering ----
function render() {{
  d3.select("#graph-panel svg").remove();
  if (!currentScenario || !currentResult) return;

  const entry = DATA.scenarios[currentScenario][currentResult];
  const graphData = entry.graph;
  const rankings  = entry.rankings;
  const threshold = parseFloat(slider.node().value);

  // Build filtered copies
  const nodes = graphData.nodes.map(d => ({{ ...d }}));
  const edges = graphData.edges
    .filter(e => (e.weight === undefined || Math.abs(e.weight) >= threshold))
    .map(d => ({{ ...d }}));

  // Importance scale (node attribute)
  const impExtent = d3.extent(nodes, d => d.importance || 0);
  const rScale = d3.scaleLinear().domain([0, impExtent[1] || 1]).range([5, 22]);

  // Edge weight scale
  const wExtent = d3.extent(edges, d => Math.abs(d.weight || 0));
  const wScale = d3.scaleLinear().domain([0, wExtent[1] || 1]).range([1, 6]);

  // Create SVG
  const panel = d3.select("#graph-panel");
  const width  = panel.node().clientWidth;
  const height = Math.max(panel.node().clientHeight, 500);

  const svg = panel.append("svg")
    .attr("viewBox", [0, 0, width, height]);

  // Arrow marker
  svg.append("defs").append("marker")
    .attr("id", "arrow").attr("viewBox", "0 -5 10 10")
    .attr("refX", 20).attr("refY", 0)
    .attr("markerWidth", 6).attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path").attr("d", "M0,-5L10,0L0,5")
    .attr("fill", "var(--green)");

  // Simulation
  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(edges).id(d => d.id).distance(120))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide().radius(d => rScale(d.importance || 0) + 8));

  // Edges
  const link = svg.append("g")
    .selectAll("line").data(edges).join("line")
    .attr("stroke", "var(--green)").attr("stroke-opacity", 0.6)
    .attr("stroke-width", d => wScale(Math.abs(d.weight || 0)))
    .attr("marker-end", "url(#arrow)");

  // Nodes
  const node = svg.append("g")
    .selectAll("circle").data(nodes).join("circle")
    .attr("r", d => rScale(d.importance || 0))
    .attr("fill", "var(--accent)").attr("stroke", "#fff").attr("stroke-width", 1.2)
    .call(drag(simulation));

  // Labels
  const label = svg.append("g")
    .selectAll("text").data(nodes).join("text")
    .attr("class", "node-label")
    .attr("dy", d => rScale(d.importance || 0) + 12)
    .text(d => d.id);

  // Tooltip events
  node.on("mouseover", function(event, d) {{
    let html = "<strong>" + d.id + "</strong>";
    if (d.importance !== undefined) html += "<br>Importance: " + d.importance.toFixed(4);
    tooltip.html(html).style("opacity", 1)
      .style("left", (event.offsetX + 12) + "px")
      .style("top", (event.offsetY - 10) + "px");
  }})
  .on("mouseout", () => tooltip.style("opacity", 0));

  link.on("mouseover", function(event, d) {{
    let html = d.source.id + " &rarr; " + d.target.id;
    if (d.weight !== undefined) html += "<br>Weight: " + d.weight.toFixed(4);
    if (d.delay  !== undefined) html += "<br>Delay: " + d.delay;
    tooltip.html(html).style("opacity", 1)
      .style("left", (event.offsetX + 12) + "px")
      .style("top", (event.offsetY - 10) + "px");
  }})
  .on("mouseout", () => tooltip.style("opacity", 0));

  simulation.on("tick", () => {{
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("cx", d => d.x).attr("cy", d => d.y);
    label.attr("x", d => d.x).attr("y", d => d.y);
  }});

  // ---- ranking table ----
  renderRankings(rankings);
}}

function renderRankings(rankings) {{
  const tbody = d3.select("#rank-table tbody");
  tbody.selectAll("tr").remove();
  if (!rankings || !rankings.length) return;

  // Ensure score is numeric
  const data = rankings.map((r, i) => ({{ name: r[0], score: +r[1], rank: i + 1 }}));

  // Sort
  data.sort((a, b) => {{
    const va = a[sortCol], vb = b[sortCol];
    return sortAsc ? d3.ascending(va, vb) : d3.descending(va, vb);
  }});
  data.forEach((d, i) => d.rank = i + 1);

  const maxScore = d3.max(data, d => d.score) || 1;

  const rows = tbody.selectAll("tr").data(data).join("tr");
  rows.append("td").text(d => d.rank);
  rows.append("td").text(d => d.name);
  rows.append("td").text(d => d.score.toFixed(4));
  rows.append("td").append("div").attr("class", "bar")
    .style("width", d => (d.score / maxScore * 100) + "%");
}}

// Column sorting
d3.selectAll("#rank-table th").on("click", function() {{
  const col = this.dataset.col;
  if (!col || col === "bar") return;
  if (sortCol === col) sortAsc = !sortAsc;
  else {{ sortCol = col; sortAsc = col === "name"; }}
  render();
}});

// Drag behaviour
function drag(simulation) {{
  return d3.drag()
    .on("start", (event, d) => {{
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    }})
    .on("drag", (event, d) => {{ d.fx = event.x; d.fy = event.y; }})
    .on("end", (event, d) => {{
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null; d.fy = null;
    }});
}}

// initial render
render();
</script>
</body>
</html>
"""


def generate_report(mode: RunModes, case: str, output_dir: str | None = None) -> Path:
    """Generate an interactive HTML report for *case*.

    Parameters
    ----------
    mode
        ``"test"`` or ``"cases"``.
    case
        The case name whose results should be included.
    output_dir
        Directory to write the HTML file into.  Defaults to
        ``<saveloc>/reports/<case>/``.

    Returns
    -------
    Path
        Absolute path of the generated HTML file.
    """
    report = ReportData(mode, case)
    data = report.collect()

    if output_dir is None:
        output_dir = os.path.join(report.saveloc, "reports", case)
    config_setup.ensure_existence(output_dir)

    data_json = json.dumps(data, default=str)
    html = _HTML_TEMPLATE.format(case=case, data_json=data_json)

    out_path = Path(output_dir) / f"{case}_report.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", out_path)
    return out_path


def generate_report_scenarios(mode: RunModes, case: str, write_output: bool) -> None:
    """Entry point matching the signature used by *run_full.py* stages."""
    if write_output:
        generate_report(mode, case)
