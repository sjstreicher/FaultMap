"""
Verifies the working of the transfer entropy calculation code by means of
an example on autoregressive data with known time delay.
"""

import jpype  # type: ignore
import pytest
from sklearn import preprocessing  # type: ignore

from faultmap.datagen import autoreg_datagen
from faultmap.infodynamics import calc_te


@pytest.fixture(scope="session", autouse=True)
def start_jvm():
    """Checks if JVM is started, and starts it if not."""
    jar_loc = "infodynamics.jar"
    # Only start the JVM if it's not already running
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jar_loc)
    yield  # This will execute before the first test

    # Teardown
    # Shut down the JVM (not strictly necessary)
    if jpype.isJVMStarted():
        jpype.shutdownJVM()


@pytest.fixture(name="generation_details")
def fixture_generation_details() -> tuple[int, int, int]:
    """Define number of samples to generate and analyse, and the delay in actual data"""
    samples = 2500
    sub_samples = 1000
    delay = 5

    return samples, sub_samples, delay


def test_peak_te_infodyn_kernel(generation_details: tuple[int, int, int]):
    """Tests that the peak transfer entropy calculated with the kernel method is at the
    correct delay."""
    samples, sub_samples, delay = generation_details
    entropies_infodyn_kernel = []
    # Calculate transfer entropies in range of +/- 5 from actual delay using
    # infodynamics package
    for time_lag in range(delay - 5, delay + 6):
        [_, x_hist, y_hist] = autoreg_datagen(delay, time_lag, samples, sub_samples)
        x_hist_norm = preprocessing.scale(x_hist, axis=1)
        y_hist_norm = preprocessing.scale(y_hist, axis=1)
        result_infodyn, _ = calc_te(
            "infodynamics.jar",
            "kernel",
            x_hist_norm[0],
            y_hist_norm[0],
            **{"kernel_width": 0.1}
        )
        entropies_infodyn_kernel.append(result_infodyn)

    maxval = max(entropies_infodyn_kernel)
    delayedval = entropies_infodyn_kernel[delay]
    assert maxval == delayedval


def test_peak_te_infodyn_kraskov(generation_details: tuple[int, int, int]):
    """Tests that the peak transfer entropy calculated with the Kraskov method is at the
    correct delay when not using auto-embedding."""
    samples, sub_samples, delay = generation_details
    entropies_infodyn_kraskov = []
    for time_lag in range(delay - 5, delay + 6):
        [_, x_hist, y_hist] = autoreg_datagen(delay, time_lag, samples, sub_samples)
        x_hist_norm = preprocessing.scale(x_hist, axis=1)
        y_hist_norm = preprocessing.scale(y_hist, axis=1)
        result_infodyn, _ = calc_te(
            "infodynamics.jar",
            "kraskov",
            x_hist_norm[0],
            y_hist_norm[0],
            test_significance=False,
            auto_embed=False,
        )
        entropies_infodyn_kraskov.append(result_infodyn)

    maxval = max(entropies_infodyn_kraskov)
    delayedval = entropies_infodyn_kraskov[delay]
    assert maxval == delayedval


def test_peak_te_kraskov_autoembed(generation_details: tuple[int, int, int]):
    """Tests that the peak transfer entropy calculated with the Kraskov method is at the
    correct delay when using auto-embedding."""
    samples, sub_samples, delay = generation_details
    entropies_infodyn_kraskov = []
    for time_lag in range(delay - 5, delay + 6):
        [_, x_hist, y_hist] = autoreg_datagen(delay, time_lag, samples, sub_samples)
        x_hist_norm = preprocessing.scale(x_hist, axis=1)
        y_hist_norm = preprocessing.scale(y_hist, axis=1)
        result_infodyn, _ = calc_te(
            "infodynamics.jar",
            "kraskov",
            x_hist_norm[0],
            y_hist_norm[0],
            test_significance=False,
            auto_embed=True,
        )
        entropies_infodyn_kraskov.append(result_infodyn)

    max_val = max(entropies_infodyn_kraskov)
    delayed_val = entropies_infodyn_kraskov[delay]
    assert max_val == delayed_val
