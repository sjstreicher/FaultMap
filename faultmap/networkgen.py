# -*- coding: utf-8 -*-
"""Generates sample digraphs for testing and demonstrating faultmap methods.

"""

import json
import os
from typing import List

import networkx as nx
import numpy as np
from datagen import seed_randn

from faultmap.data_processing import build_graph

files_loc = json.load(open("test/testconfig.json", encoding="utf-8"))
save_loc = os.path.expanduser(files_loc["save_loc"])

seed_list = [35, 88, 107, 52, 98]


def numbered_variables(name: str, n_vars: int) -> List[str]:
    """

    Args:
        name:
        n_vars:

    Returns:

    """
    return [f"{name} {index + 1}" for index in range(n_vars)]


def graph_name(name: str):
    """

    Args:
        name:

    Returns:

    """
    return os.path.join(save_loc, "test_graphs", name + ".gml")


all_test_functions = []


def gen_random_array(dimension):
    """Generates square normally distributed random array."""
    seeds = iter(seed_list)
    random_array = np.expand_dims(
        seed_randn(next(seeds), dimension**2), axis=1
    ).reshape((dimension, dimension))

    return np.abs(random_array)


# TODO: Replace this with a proper class
def test_function_builder(
    connections, gain_matrix=None, filename=None, variables=None, doc=None
):
    n_vars, _ = connections.shape

    if gain_matrix is None:
        gain_matrix = connections.copy()

    if variables is None:
        variables = numbered_variables("X", n_vars)

    def graph_generator(draw=False):
        test_graph = build_graph(
            variables, gain_matrix, connections, np.ones(len(variables))
        )

        if draw and filename:
            nx.write_gml(test_graph, filename)
            nx.draw(test_graph)

        return connections, gain_matrix, variables, test_graph

    if doc:
        graph_generator.__doc__ = doc
    all_test_functions.append(graph_generator)
    return graph_generator


fullconn_equal = test_function_builder(
    np.ones((5, 5)),
    filename="fullcon_equal",
    doc=(
        """Creates a fully connected 5x5 digraph with unit weights """
        + """on all edges."""
    ),
)

fullconn_random = test_function_builder(
    gen_random_array(5),
    filename="fullconn_randn",
    doc="""Creates a fully connected 5x5 digraph with normally
        distributed positive weights on all edges.""",
)

series_equal_five = test_function_builder(
    np.diag(np.ones(4), -1), filename="series_equal_5"
)
series_equal_four = test_function_builder(
    np.diag(np.ones(3), -1), filename="series_equal_4"
)
series_equal_three = test_function_builder(
    np.diag(np.ones(2), -1), filename="series_equal_3"
)

connect_5 = np.diag([1, 1, 1, 1, 0], -1)
connect_2nd = np.diag([0, 1], 4)
connect_3rd = np.diag([0, 0, 1], 3)
connect_4th = np.diag([0, 0, 0, 1], 2)

series_incomingon2nd = test_function_builder(
    connect_5 + connect_2nd,
    variables=numbered_variables("X", 5) + ["I 1"],
    filename="series_incomingon2nd",
)

series_incomingon3rd = test_function_builder(
    connect_5 + connect_3rd,
    variables=numbered_variables("X", 5) + ["I 1"],
    filename="series_incomingon3rd",
)

series_incomingon2ndand3rd = test_function_builder(
    connect_5 + connect_2nd + connect_3rd,
    variables=numbered_variables("X", 5) + ["I 1"],
    filename="series_incomingon2ndand3rd",
)

series_incomingon2ndand4th = test_function_builder(
    connect_5 + connect_2nd + connect_4th,
    variables=numbered_variables("X", 5) + ["I 1"],
    filename="series_incomingon2ndand4th",
)

series_disjoint_equal = test_function_builder(
    np.diag([1, 1, 0, 1, 1], -1),
    variables=(numbered_variables("X", 3) + numbered_variables("Y", 3)),
    filename="series_disjoint_equal",
    doc=(
        """Creates two sets of three tags connected in series """
        + """with unit weights on edges."""
    ),
)

series_disjoint_unequal = test_function_builder(
    np.diag([1, 1, 0, 1, 1], -1),
    np.diag([1, 1, 0, 2, 2], -1),
    variables=(numbered_variables("X", 3) + numbered_variables("Y", 3)),
    filename="series_disjoint_unequal",
    doc="""Creates two sets of three tags connected in series with unit
        weights on the edges of one series and weights of 2 on that
        of the other.""",
)

series_disjoint_unequalsource = test_function_builder(
    (np.diag([1, 1, 0, 1, 1, 0], -1) + np.diag([1], 6) + np.diag([0, 0, 0, 1], 3)),
    (np.diag([1, 1, 0, 1, 1, 0], -1) + np.diag([1], 6) + np.diag([0, 0, 0, 2], 3)),
    variables=(numbered_variables("X", 3) + numbered_variables("Y", 3) + ["I 1"]),
    filename="series_disjoint_unequalsource",
    doc="""Creates two sets of three tags connected in series with unit weights
        on the edges of one series and weights of 2 on that of the other.""",
)

circle_equal = test_function_builder(
    np.diag([1, 1, 1, 1], -1) + np.diag([1], 4),
    filename="circle_equal",
    doc="""Creates five tags connected in a circle with unit weights on
        all edges.""",
)

circle_unequalon2to3 = test_function_builder(
    np.diag([1, 1, 1, 1], -1) + np.diag([1], 4),
    np.diag([1, 2, 1, 1], -1) + np.diag([1], 4),
    filename="circle_unequalon2_to3",
    doc="""Creates five tags connected in a circle with unit weights on
        all edges except one.""",
)


def bias_thirdlarger():
    return np.array([1, 1, 3, 1, 1]), None
