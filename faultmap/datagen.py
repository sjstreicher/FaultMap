"""Generates various test and demo data sets.

"""
from functools import partial
from typing import Callable

import control  # type: ignore
import numpy as np
from numpy import vstack

import faultmap.data_processing

seed_list = [35, 88, 107, 52, 98]


def connection_matrix_maker(n_dims: int) -> Callable[[], tuple[list[str], np.array]]:
    """

    Args:
        n_dims:

    Returns:

    """

    def maker():
        variables = [f"X {index}" for index in range(1, n_dims + 1)]
        connection_matrix = np.ones((n_dims, n_dims), dtype=int)
        return variables, connection_matrix

    maker.__doc__ = f"Generates a {n_dims}x{n_dims} connection matrix for use in test."
    return maker


connectionmatrix_2x2, connectionmatrix_4x4, connectionmatrix_5x5 = [
    connection_matrix_maker(n) for n in [2, 4, 5]
]


def seed_random(method, seed, samples):
    """Set random seed."""
    np.random.seed(int(seed))
    return method(int(samples))


# Normal distribution
seed_randn = partial(seed_random, np.random.randn)
# Uniform distribution over [0, 1)
seed_rand = partial(seed_random, np.random.rand)


def autoreg_gen(params):
    """Generates an autoregressive set of vectors.

    A constant seed is used for testing comparison purposes.

    """

    samples = params[0]
    delay = params[1]
    if len(params) >= 3:
        alpha = params[2]
    else:
        alpha = None
    if len(params) == 4:
        noise_ratio = params[3]
    else:
        noise_ratio = None

    # Define seed for initial source data
    seeds = iter(seed_list)
    cause = seed_randn(next(seeds), samples + delay)
    affected = np.zeros_like(cause)
    # Very close covariance occasionally breaks the kde estimator
    # Another small random element is added to take care of this
    # This is not expected to be a problem on any "real" data

    # Define seed for noise data
    affected_random_add = seed_rand(next(seeds), samples + delay) - 0.5

    for i in range(delay, len(cause)):
        if alpha is None:
            affected[i] = affected[i - 1] + cause[i - (delay + 1)]
        else:
            affected[i] = alpha * affected[i - 1] + (1 - alpha) * cause[i - delay]

    affected = affected[delay:]
    cause = cause[delay:]

    if noise_ratio is not None:
        affected = affected + (affected_random_add[delay:] * noise_ratio)

    data = vstack([cause, affected])

    return data.T


def delay_gen(params):
    """Generates a normally distributed random data vector
    and a pure delay companion.

    Parameters
    ----------
        params : list
            List with the first entry being the sample length of the returned
            signals and the second entry the delay between them.

    Returns
    -------
        data : numpy.ndarray
            Array containing the generated signals arranged in columns.

    """

    samples = params[0]
    delay = params[1]

    # Define seed for initial source data
    seeds = iter(seed_list)
    cause = seed_randn(next(seeds), samples + delay)
    affected = np.zeros_like(cause)
    # Very close covariance occasionally breaks the kde estimator
    # Another small random element is added to take care of this
    # This is not expected to be a problem on any "real" data

    # Define seed for noise data
    affected_random_add = seed_rand(next(seeds), samples + delay) - 0.5

    for i in range(delay, len(cause)):
        affected[i] = cause[i - delay]

    affected = affected[delay:]
    cause = cause[delay:]

    affected = affected + affected_random_add[delay:]

    data = vstack([cause, affected])

    return data.T


def random_gen(params, n=2):
    """Generates n independent random data vectors"""

    samples = params[0]

    assert n < len(seed_list), "Not enough seeds in seed_list"
    data = vstack([seed_randn(seed, samples) for seed in seed_list[:n]])

    return data.T


def autoreg_datagen(delay, timelag, samples, sub_samples, k=1, l=1):
    """Generates autoreg data for a specific timelag (equal to
    prediction horizon) for a set of autoregressive data.

    sub_samples is the amount of samples in the dataset used to calculate the
    transfer entropy between two vectors (taken from the end of the dataset).
    sub_samples <= samples

    Currently only supports k = 1; l = 1

    You can search through a set of time lags in an attempt to identify the
    original delay.
    The transfer entropy should have a maximum value when timelag = delay
    used to generate the autoregressive dataset.

    """

    params = [samples, delay]

    data = autoreg_gen(params).T

    [x_pred, x_hist, y_hist] = faultmap.data_processing.vectorselection(
        data, timelag, sub_samples, k, l
    )

    return x_pred, x_hist, y_hist


def sinusoid_shift_gen(
    params, period=100, noise_amplitude=0.1, n_signals=5, add_noise=False
):
    """Generates sinusoid signals together with optionally uniform noise.
    The signals are shifted by a quarter period.

    Parameters
    ----------
        params : list
            List with the first (and only) entry being the sample length of
            the returned signals.
        period : int, default=100
            The period of the sinusoid in terms of samples.
        noise_amplitude : float, default=0.5
           A multiplier for mean-centred uniform noise to be added to the
           signal. The amplitude of the sine is unity.
        n_signals : int, default=5
            How many signals to return.
        add_noise : bool, default=False
            If True, noise is added to the sinusoidal signals.

    Returns
    -------
        data : numpy.ndarray
            Array containing the generated signals arranged in columns.

    """

    samples = params[0]

    frequency = 1.0 / period

    timespan = range(samples + 2 * period)

    # Generate source sine curve
    sine = [np.sin(frequency * t * 2 * np.pi) for t in timespan]

    if add_noise:
        sine_noise = (seed_rand(117, len(timespan))) - 0.5 * noise_amplitude

        sine += sine_noise

    vectors = []

    for i in range(n_signals):
        sample_shift = (period / 4) * i
        vectors.append(sine[sample_shift : samples + sample_shift])

    data = vstack(vectors)

    return data.T


def sinusoid_gen(params, period=100, noise_amplitude=1.0):
    """Generates sinusoid signals together with optionally uniform noise.
    The signals are shifted by a quarter period.

    Parameters
    ----------
        params : list
            List with the first (and only) entry being the sample length of
            the returned signals.
        period : int, default=100
            The period of the sinusoid in terms of samples.
        noise_amplitude : float, default=0.5
           A multiplier for mean-centred uniform noise to be added to the
           signal. The amplitude of the sine is unity.

    Returns
    -------
        data : numpy.ndarray
            Array containing the generated signals arranged in columns.

    """

    samples = params[0]
    delay = params[1]

    timespan = range(samples + delay)
    frequency = 1.0 / period
    cause = [np.sin(frequency * t * 2 * np.pi) for t in timespan]

    affected = np.zeros_like(cause)

    cause_closed = np.zeros_like(cause)

    for i in range(delay, len(cause)):
        affected[i] = cause[i - delay]

    affected_random_add = (seed_rand(117, samples + delay) - 0.5) * noise_amplitude

    affected += affected_random_add

    for i in range(delay, len(cause)):
        cause_closed[i] = affected[i] + cause[i]

    affected = affected[delay:]
    cause = cause[delay:]
    cause_closed = cause_closed[delay:]

    return timespan[:-delay], cause, affected, cause_closed


def firstorder_gen(params, period=0.01, noiseamp=1.0):
    """Simple first order transfer function affected variable
    with sinusoid cause.

    """

    samples = params[0]
    delay = params[1]

    process = control.matlab.tf([10], [100, 1])

    time_points = np.array(range(samples + delay))

    sine_input = np.array([np.sin(period * t * 2 * np.pi) for t in time_points])

    process_response = control.matlab.lsim(process, sine_input, time_points)

    affected_random_add = (seed_rand(51, samples + delay) - 0.5) * noiseamp

    cause = sine_input[:samples]

    if delay == 0:
        offset = None
    else:
        offset = samples

    affected = process_response[0][delay:] + affected_random_add[delay:]

    tspan = process_response[1][:offset]

    return tspan, cause, affected
