"""Unit tests for datagen module."""

import numpy as np

from faultmap.datagen import (
    autoreg_gen,
    connection_matrix_maker,
    delay_gen,
    random_gen,
)


class TestConnectionMatrixMaker:
    def test_2x2(self):
        maker = connection_matrix_maker(2)
        variables, matrix = maker()
        assert len(variables) == 2
        assert matrix.shape == (2, 2)
        assert variables == ["X 1", "X 2"]
        np.testing.assert_array_equal(matrix, np.ones((2, 2), dtype=int))

    def test_5x5(self):
        maker = connection_matrix_maker(5)
        variables, matrix = maker()
        assert len(variables) == 5
        assert matrix.shape == (5, 5)


class TestAutoregGen:
    def test_output_shape(self):
        params = [100, 5]
        data = autoreg_gen(params)
        assert data.shape == (100, 2)

    def test_deterministic_with_seed(self):
        params = [50, 3]
        data1 = autoreg_gen(params)
        data2 = autoreg_gen(params)
        np.testing.assert_array_equal(data1, data2)

    def test_with_alpha(self):
        params = [100, 5, 0.5]
        data = autoreg_gen(params)
        assert data.shape == (100, 2)

    def test_with_noise(self):
        params = [100, 5, 0.5, 0.1]
        data = autoreg_gen(params)
        assert data.shape == (100, 2)


class TestDelayGen:
    def test_output_shape(self):
        params = [100, 5]
        data = delay_gen(params)
        assert data.shape == (100, 2)

    def test_deterministic(self):
        params = [50, 3]
        data1 = delay_gen(params)
        data2 = delay_gen(params)
        np.testing.assert_array_equal(data1, data2)


class TestRandomGen:
    def test_output_shape(self):
        params = [100]
        data = random_gen(params, n=3)
        assert data.shape == (100, 3)

    def test_default_two_columns(self):
        params = [50]
        data = random_gen(params)
        assert data.shape == (50, 2)
