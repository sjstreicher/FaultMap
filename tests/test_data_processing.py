"""Unit tests for data_processing module.

Tests pure Python functions that don't require Java/JIDT.
"""

import csv
import os
import tempfile

import networkx as nx
import numpy as np
import pytest

from faultmap.data_processing import (
    build_graph,
    detrend_first_differences,
    detrend_linear_model,
    detrend_link_relatives,
    get_folders,
    normalise_data,
    read_connectionmatrix,
    read_header_values_datafile,
    read_matrix,
    read_timestamps,
    read_variables,
    shuffle_data,
    skogestad_scale_select,
    split_time_series_data,
    subtract_mean,
    vectorselection,
    writecsv,
)


class TestShuffleData:
    def test_output_shape(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = shuffle_data(data)
        assert result.shape == (1, 5)

    def test_preserves_values(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = shuffle_data(data)
        assert sorted(result[0]) == sorted(data)


class TestGetFolders:
    def test_simple_path(self):
        result = get_folders("/home/user/data")
        assert result == ["/", "home", "user", "data"]

    def test_relative_path(self):
        result = get_folders("foo/bar/baz")
        assert result == ["foo", "bar", "baz"]


class TestSplitTimeSeriesData:
    def test_single_box(self):
        data = np.arange(100)
        boxes = split_time_series_data(data, 1.0, 100, 1)
        assert len(boxes) == 1
        np.testing.assert_array_equal(boxes[0], data)

    def test_multiple_boxes(self):
        data = np.arange(100)
        boxes = split_time_series_data(data, 1.0, 50, 2)
        assert len(boxes) == 2
        assert len(boxes[0]) == 50
        assert len(boxes[1]) == 50

    def test_box_count(self):
        data = np.arange(200)
        boxes = split_time_series_data(data, 1.0, 50, 3)
        assert len(boxes) == 3


class TestSubtractMean:
    def test_mean_subtracted(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = subtract_mean(data)
        for col in range(data.shape[1]):
            assert abs(np.mean(result[:, col])) < 1e-10

    def test_shape_preserved(self):
        data = np.random.randn(10, 3)
        result = subtract_mean(data)
        assert result.shape == data.shape


class TestSkogestadScaleSelect:
    def test_disturbance_variable(self):
        result = skogestad_scale_select("D", 0.0, 5.0, 8.0)
        assert result == 5.0  # max(5-0, 8-5) = max(5, 3)

    def test_state_variable(self):
        result = skogestad_scale_select("S", 0.0, 5.0, 8.0)
        assert result == 3.0  # min(5-0, 8-5) = min(5, 3)

    def test_invalid_type_raises_valueerror(self):
        with pytest.raises(ValueError, match="Variable type flag not recognized"):
            skogestad_scale_select("X", 0.0, 5.0, 8.0)


class TestDetrendMethods:
    def test_first_differences(self):
        data = np.array([[1.0, 10.0], [3.0, 12.0], [6.0, 15.0]])
        result = detrend_first_differences(data)
        assert result.shape == data.shape
        # First row should be zero
        np.testing.assert_array_equal(result[0], [0.0, 0.0])

    def test_linear_model(self):
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = detrend_linear_model(data)
        # Perfectly linear data should detrend to near-zero
        assert np.max(np.abs(result)) < 1e-10

    def test_link_relatives(self):
        data = np.array([[10.0], [11.0], [12.1]])
        result = detrend_link_relatives(data)
        assert result.shape == data.shape
        # First row should be zero
        assert result[0, 0] == 0.0


class TestBuildGraph:
    def test_simple_graph(self):
        variables = ["A", "B", "C"]
        gain_matrix = np.array([[0, 1, 0], [0, 0, 2], [0, 0, 0]])
        connections = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        bias_vector = np.array([1.0, 1.0, 1.0])

        graph = build_graph(variables, gain_matrix, connections, bias_vector)

        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes()) == 3
        assert len(graph.edges()) == 2

    def test_node_bias(self):
        variables = ["A", "B"]
        gain_matrix = np.array([[0, 1], [0, 0]])
        connections = np.array([[0, 1], [0, 0]])
        bias_vector = np.array([2.0, 3.0])

        graph = build_graph(variables, gain_matrix, connections, bias_vector)

        assert graph.nodes["A"]["bias"] == 2.0
        assert graph.nodes["B"]["bias"] == 3.0


class TestVectorSelection:
    def test_basic_selection(self):
        data = np.array(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
            dtype=float,
        )
        x_pred, x_hist, y_hist = vectorselection(data, timelag=1, sub_samples=5)

        assert x_pred.shape == (1, 5)
        assert x_hist.shape == (1, 5)
        assert y_hist.shape == (1, 5)


class TestCSVReadWrite:
    def test_writecsv_and_read(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            tmpfile = f.name

        try:
            header = ["Time", "Var1", "Var2"]
            items = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            writecsv(tmpfile, items, header)

            values, read_header = read_header_values_datafile(tmpfile)
            assert read_header == header
            assert values.shape == (2, 3)
            np.testing.assert_array_almost_equal(values[0], [1.0, 2.0, 3.0])
        finally:
            os.unlink(tmpfile)

    def test_read_variables(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Var1", "Var2", "Var3"])
            writer.writerow([0, 1.0, 2.0, 3.0])
            tmpfile = f.name

        try:
            variables = read_variables(tmpfile)
            assert variables == ["Var1", "Var2", "Var3"]
        finally:
            os.unlink(tmpfile)

    def test_read_timestamps(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Var1"])
            writer.writerow([100, 1.0])
            writer.writerow([200, 2.0])
            tmpfile = f.name

        try:
            timestamps = read_timestamps(tmpfile)
            assert len(timestamps) == 2
            assert timestamps[0] == "100"
            assert timestamps[1] == "200"
        finally:
            os.unlink(tmpfile)

    def test_read_connectionmatrix(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["", "A", "B"])
            writer.writerow(["A", 0, 1])
            writer.writerow(["B", 1, 0])
            tmpfile = f.name

        try:
            matrix, variables = read_connectionmatrix(tmpfile)
            assert variables == ["A", "B"]
            assert matrix.shape == (2, 2)
            assert matrix[0, 1] == 1.0
        finally:
            os.unlink(tmpfile)

    def test_read_matrix(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["", "A", "B"])
            writer.writerow(["A", 1.0, 2.0])
            writer.writerow(["B", 3.0, 4.0])
            tmpfile = f.name

        try:
            matrix = read_matrix(tmpfile)
            assert matrix.shape == (2, 2)
            np.testing.assert_array_almost_equal(matrix, [[1.0, 2.0], [3.0, 4.0]])
        finally:
            os.unlink(tmpfile)


class TestNormaliseData:
    def test_standardise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            headerline = ["Time", "Var1", "Var2"]
            timestamps = np.array([0, 1, 2, 3, 4], dtype=float)
            inputdata = np.array(
                [[10, 20], [12, 22], [14, 24], [16, 26], [18, 28]], dtype=float
            )
            variables = ["Var1", "Var2"]

            result = normalise_data(
                headerline,
                timestamps,
                inputdata,
                variables,
                tmpdir,
                "testcase",
                "testscenario",
                "standardise",
                None,
            )
            # Standardised data should have zero mean per column
            for col in range(result.shape[1]):
                assert abs(np.mean(result[:, col])) < 1e-10

    def test_no_normalisation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            headerline = ["Time", "Var1"]
            timestamps = np.array([0, 1, 2], dtype=float)
            inputdata = np.array([[1], [2], [3]], dtype=float)
            variables = ["Var1"]

            result = normalise_data(
                headerline,
                timestamps,
                inputdata,
                variables,
                tmpdir,
                "testcase",
                "testscenario",
                False,
                None,
            )
            np.testing.assert_array_equal(result, inputdata)

    def test_invalid_method_raises_valueerror(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Normalisation method not recognized"):
                normalise_data(
                    ["Time", "V"],
                    np.array([0.0]),
                    np.array([[1.0]]),
                    ["V"],
                    tmpdir,
                    "c",
                    "s",
                    "invalid_method",
                    None,
                )
