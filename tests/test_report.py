"""Unit tests for the report module."""

import csv

import networkx as nx

from faultmap.report import (
    ReportData,
    _collect_scenario,
    _read_graph,
    _read_ranking_csv,
    generate_report,
)


class TestReadGraph:
    def test_reads_valid_gml(self, tmp_path):
        g = nx.DiGraph()
        g.add_node("A", importance=0.6)
        g.add_node("B", importance=0.4)
        g.add_edge("A", "B", weight=0.8, delay=2.0)
        gml_path = str(tmp_path / "test.gml")
        nx.write_gml(g, gml_path)

        result = _read_graph(gml_path)
        assert result is not None
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1

        node_ids = {n["id"] for n in result["nodes"]}
        assert node_ids == {"A", "B"}

        edge = result["edges"][0]
        assert edge["source"] == "A"
        assert edge["target"] == "B"
        assert abs(edge["weight"] - 0.8) < 1e-10
        assert abs(edge["delay"] - 2.0) < 1e-10

    def test_returns_none_for_bad_file(self, tmp_path):
        bad_path = str(tmp_path / "nonexistent.gml")
        assert _read_graph(bad_path) is None

    def test_reads_node_importance(self, tmp_path):
        g = nx.DiGraph()
        g.add_node("X1", importance=0.9)
        g.add_node("X2", importance=0.1)
        gml_path = str(tmp_path / "imp.gml")
        nx.write_gml(g, gml_path)

        result = _read_graph(gml_path)
        imps = {n["id"]: n["importance"] for n in result["nodes"]}
        assert abs(imps["X1"] - 0.9) < 1e-10
        assert abs(imps["X2"] - 0.1) < 1e-10


class TestReadRankingCsv:
    def test_reads_ranking_csv(self, tmp_path):
        csv_path = str(tmp_path / "rankinglist_eigenvector.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["NodeA", "0.6"])
            writer.writerow(["NodeB", "0.3"])
            writer.writerow(["NodeC", "0.1"])

        result = _read_ranking_csv(csv_path)
        assert len(result) == 3
        assert result[0] == ["NodeA", 0.6]
        assert result[1] == ["NodeB", 0.3]

    def test_returns_empty_for_missing_file(self):
        result = _read_ranking_csv("/nonexistent/path.csv")
        assert result == []


class TestCollectScenario:
    def test_collects_gml_and_csv(self, tmp_path):
        # Mimick noderank output hierarchy
        box_dir = (
            tmp_path / "te_kraskov" / "nosig"
            / "noembed" / "weight" / "box001"
            / "nodummies"
        )
        box_dir.mkdir(parents=True)

        g = nx.DiGraph()
        g.add_node("V1", importance=0.7)
        g.add_node("V2", importance=0.3)
        g.add_edge("V1", "V2", weight=0.5, delay=1.0)
        nx.write_gml(g, str(box_dir / "graph_eigenvector.gml"))

        with open(
            box_dir / "rankinglist_eigenvector.csv",
            "w",
            newline="",
        ) as f:
            csv.writer(f).writerows(
                [["V1", "0.7"], ["V2", "0.3"]]
            )

        entries = _collect_scenario(str(tmp_path))
        assert len(entries) >= 1

        key = [k for k in entries if "eigenvector" in k][0]
        entry = entries[key]
        assert len(entry["graph"]["nodes"]) == 2
        assert len(entry["graph"]["edges"]) == 1
        assert len(entry["rankings"]) == 2

    def test_empty_dir_returns_empty(self, tmp_path):
        entries = _collect_scenario(str(tmp_path))
        assert entries == {}


class TestGenerateReport:
    def test_generates_html_file(self, tmp_path):
        """Test report generation with synthetic noderank data."""
        saveloc = tmp_path / "results"
        noderank_dir = (
            saveloc / "noderank" / "testcase" / "scenario1"
        )
        box_dir = (
            noderank_dir / "method" / "sig" / "embed"
            / "weight" / "box001" / "nodummies"
        )
        box_dir.mkdir(parents=True)

        g = nx.DiGraph()
        g.add_node("A", importance=1.0)
        g.add_node("B", importance=0.5)
        g.add_edge("A", "B", weight=0.9, delay=1.0)
        nx.write_gml(g, str(box_dir / "graph_eigenvector.gml"))

        with open(
            box_dir / "rankinglist_eigenvector.csv",
            "w",
            newline="",
        ) as f:
            csv.writer(f).writerows(
                [["A", "1.0"], ["B", "0.5"]]
            )

        # Verify collection works on raw ReportData
        report = ReportData.__new__(ReportData)
        report.saveloc = str(saveloc)
        report.case = "testcase"
        report.mode = "test"

        data = report.collect()
        assert "scenario1" in data["scenarios"]

        # Generate the full HTML with patched config
        import faultmap.report as report_mod

        original = report_mod.config_setup.run_setup
        report_mod.config_setup.run_setup = lambda m, c: (
            str(saveloc),
            str(tmp_path / "config"),
            str(tmp_path / "data"),
            "",
        )
        try:
            output_dir = str(tmp_path / "output")
            path = generate_report(
                "test", "testcase", output_dir=output_dir
            )
        finally:
            report_mod.config_setup.run_setup = original

        assert path.exists()
        content = path.read_text()
        assert "FaultMap Interactive Report" in content
        assert "testcase" in content
        assert '"nodes"' in content
        assert '"edges"' in content
        assert "d3.v7.min.js" in content
