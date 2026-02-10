"""Unit tests for noderank module."""

import numpy as np
import pytest

from faultmap.noderank import (
    calc_transient_importancediffs,
    gainmatrix_preprocessing,
    gainmatrix_tobinary,
    norm_dict,
    normalise_rankinglist,
)


class TestNormDict:
    def test_normalises_values(self):
        d = {"a": 2.0, "b": 3.0, "c": 5.0}
        result = norm_dict(d)
        assert abs(sum(result.values()) - 1.0) < 1e-10
        assert abs(result["a"] - 0.2) < 1e-10
        assert abs(result["b"] - 0.3) < 1e-10
        assert abs(result["c"] - 0.5) < 1e-10


class TestGainmatrixToBinary:
    def test_converts_to_binary(self):
        gainmatrix = np.array([[0.0, 1.5, 0.0], [2.3, 0.0, 0.0], [0.0, 0.7, 0.0]])
        result = gainmatrix_tobinary(gainmatrix)
        expected = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        np.testing.assert_array_equal(result, expected)

    def test_zero_matrix_stays_zero(self):
        gainmatrix = np.zeros((3, 3))
        result = gainmatrix_tobinary(gainmatrix)
        np.testing.assert_array_equal(result, np.zeros((3, 3)))


class TestGainmatrixPreprocessing:
    def test_mean_scaling(self):
        gainmatrix = np.array([[0.0, 2.0], [4.0, 0.0]])
        result, currentmean = gainmatrix_preprocessing(gainmatrix)
        assert abs(currentmean - 3.0) < 1e-10
        # After scaling, mean of nonzero elements should be 1.0
        nonzero_elements = result[result.nonzero()]
        assert abs(np.mean(nonzero_elements) - 1.0) < 1e-10


class TestCalcTransientImportanceDiffs:
    def test_single_dict(self):
        rankingdicts = [{"a": 0.5, "b": 0.3, "c": 0.2}]
        variablelist = ["a", "b", "c"]
        transient, baseval, boxrank, rel_boxrank = calc_transient_importancediffs(
            rankingdicts, variablelist
        )
        assert baseval["a"] == 0.5
        assert len(transient["a"]) == 0  # No differences with single dict
        assert boxrank["a"] == [0.5]

    def test_two_dicts(self):
        rankingdicts = [
            {"a": 0.5, "b": 0.3, "c": 0.2},
            {"a": 0.4, "b": 0.4, "c": 0.2},
        ]
        variablelist = ["a", "b", "c"]
        transient, baseval, boxrank, rel_boxrank = calc_transient_importancediffs(
            rankingdicts, variablelist
        )
        assert abs(transient["a"][0] - (-0.1)) < 1e-10
        assert abs(transient["b"][0] - 0.1) < 1e-10
        assert abs(transient["c"][0] - 0.0) < 1e-10
        assert len(boxrank["a"]) == 2


class TestNormaliseRankinglist:
    def test_normalisation(self):
        rankingdict = {"a": 0.5, "b": 0.3, "c": 0.2, "dummy": 0.1}
        original_vars = ["a", "b", "c"]
        result = normalise_rankinglist(rankingdict, original_vars)
        # Result is a sorted list of tuples
        total = sum(v for _, v in result)
        assert abs(total - 1.0) < 1e-10
        # Should be sorted in descending order
        values = [v for _, v in result]
        assert values == sorted(values, reverse=True)
