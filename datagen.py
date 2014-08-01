"""
Created on Mon Feb 24 15:27:21 2014

@author: Simon Streicher
"""

from numpy import vstack
import numpy as np

import control

from transentropy import vectorselection


def connectionmatrix_maker(N):
    def maker():
        variables = ['X {}'.format(i) for i in range(1, N+1)]
        connectionmatrix = np.ones((N, N), dtype=int)
        return variables, connectionmatrix
    maker.__doc__  = """ Generates a {0}x{0} connection matrix for use in tests.""".format(N)
    return maker

connectionmatrix_2x2, connectionmatrix_4x4, connectionmatrix_5x5 = [
    connectionmatrix_maker(N) for N in [2, 4, 5]
    ]


def autoreg_gen(samples, delay):
    """Generates an autoregressive set of vectors.

    A constant seed is used for testing comparison purposes.

    """

    # Define seed for initial source data
    np.random.seed(35)

    cause = np.random.randn(samples + delay)
    affected = np.zeros_like(cause)
    # Very close covariance occassionally breaks the kde estimator
    # Another small random element is added to take care of this
    # This is not expected to be a problem on any "real" data

    # Define seed for noise data
    np.random.seed(88)
    affected_random_add = np.random.rand(samples + delay)

    for i in range(delay, len(cause)):
        affected[i] = affected[i - 1] + cause[i - delay]

    affected = affected[delay:]
    cause = cause[delay:]

    affected = affected + affected_random_add[delay:]

    data = vstack([cause, affected])

    return data.T


def delay_gen(samples, delay):
    """Generates a random data vector and a pure delay companion.

    A constant seed is used for testing comparison purposes.

    """

    # Define seed for initial source data
    np.random.seed(35)

    cause = np.random.randn(samples + delay)
    affected = np.zeros_like(cause)
    # Very close covariance occassionally breaks the kde estimator
    # Another small random element is added to take care of this
    # This is not expected to be a problem on any "real" data

    # Define seed for noise data
    np.random.seed(88)
    affected_random_add = np.random.rand(samples + delay)

    for i in range(delay, len(cause)):
        affected[i] = cause[i - delay]

    affected = affected[delay:]
    cause = cause[delay:]

    affected = affected + affected_random_add[delay:]

    data = vstack([cause, affected])

    return data.T


def random_gen(samples, delay):
    """Generates two completely independent random data vectors."""

    # Generate first vector
    np.random.seed(35)
    x1 = np.random.randn(samples)

    # Generate second vector
    np.random.seed(88)
    x2 = np.random.randn(samples)

    data = vstack([x1, x2])

    return data.T


def random_gen_5x5(samples, delay):
    """Generates five completely independent random data vectors.


    """

    # Generate first vector
    np.random.seed(35)
    x1 = np.random.randn(samples)

    # Generate second vector
    np.random.seed(88)
    x2 = np.random.randn(samples)

    # Generate third vector
    np.random.seed(107)
    x3 = np.random.randn(samples)

    # Generate fourth vector
    np.random.seed(52)
    x4 = np.random.randn(samples)

    # Generate fifth vector
    np.random.seed(98)
    x5 = np.random.randn(samples)

    data = vstack([x1, x2, x3, x4, x5])

    return data.T


def autoreg_datagen(delay, timelag, samples, sub_samples, k=1, l=1):
    """Generates autoreg data for a specific timelag (equal to
    prediction horison) for a set of autoregressive data.

    sub_samples is the amount of samples in the dataset used to calculate the
    transfer entropy between two vectors (taken from the end of the dataset).
    sub_samples <= samples

    Currently only supports k = 1; l = 1

    You can search through a set of timelags in an attempt to identify the
    original delay.
    The transfer entropy should have a maximum value when timelag = delay
    used to generate the autoregressive dataset.

    """

    data = autoreg_gen(samples, delay).T

    [x_pred, x_hist, y_hist] = vectorselection(data, timelag,
                                               sub_samples, k, l)

    return x_pred, x_hist, y_hist


def sinusoid_shift_gen(samples, delay, period=100, noiseamp=0.1,
                       addnoise=False):
    """Generates a sinusoid, with delayed noise companion
    and a closed loop sinusoid with delay and noise.

    period is the number of samples for each cycle

    noiseamp is the maximum amplitude of the noise added to the signal

    """

    frequency = 1./period

    tspan = range(samples + 2*period)

    # Generate source sine curve
    sine = [np.sin(frequency * t*2*np.pi) for t in tspan]

    if addnoise:
        np.random.seed(117)
        sine_noise = (np.random.randn(len(tspan)) - 0.5) * noiseamp

        sine = sine + sine_noise

    # First vector is simply the first samples of the sine vector
    x1 = sine[0:samples]

    # Define the second vector
    sampleshift = (period/4)*1
    x2 = sine[sampleshift:samples+sampleshift]

    # Third vector
    sampleshift = (period/4)*2
    x3 = sine[sampleshift:samples+sampleshift]

    # Fourth vector
    sampleshift = (period/4)*3
    x4 = sine[sampleshift:samples+sampleshift]

    # Fifth vector
    # The fifth vector is the same except for any noise added to the sine
    sampleshift = (period/4)*4
    x5 = sine[sampleshift:samples+sampleshift]

    data = vstack([x1, x2, x3, x4, x5])

    return data.T


def sinusoid_gen(samples, delay, period=0.01, noiseamp=1.0):
    """Generates four sinusoids, each based on the same frequency but differing
    in phase by 90 degrees.

    period is the number of cycles for each sample

    noiseamp is the standard deviation of the noise added to the signal

    """

    tspan = range(samples + delay)

    cause = [np.sin(period * t*2*np.pi) for t in tspan]

    affected = np.zeros_like(cause)

    cause_closed = np.zeros_like(cause)

    for i in range(delay, len(cause)):
        affected[i] = cause[i - delay]

    np.random.seed(117)
    affected_random_add = (np.random.rand(samples + delay) - 0.5) * noiseamp

    affected = affected + affected_random_add

    for i in range(delay, len(cause)):
        cause_closed[i] = affected[i] + cause[i]

    affected = affected[delay:]
    cause = cause[delay:]
    cause_closed = cause_closed[delay:]

    return tspan[:-delay], cause, affected, cause_closed


def firstorder_gen(samples, delay, period=0.01, noiseamp=1.0):
    """Simple first order transfer function affected variable
    with sinusoid cause.

    """

    P1 = control.matlab.tf([10], [100, 1])

    timepoints = np.array(range(samples + delay))

    sine_input = np.array([np.sin(period * t*2*np.pi) for t in timepoints])

    P1_response = control.matlab.lsim(P1, sine_input, timepoints)

    np.random.seed(51)
    affected_random_add = (np.random.rand(samples + delay) - 0.5) * noiseamp

    cause = sine_input[:samples]

    if delay == 0:
        offset = None
    else:
        offset = samples

    affected = P1_response[0][delay:] + affected_random_add[delay:]

    tspan = P1_response[1][:offset]

    return tspan, cause, affected


def oscillating_feedback_5x5(samples, delays=[3, 2, 5, 4], period=0.01,
                             noiseamp=1.0):
    """Passes sine source signal through a number of transfer functions
    before connecting back on itself.

    delays is a list of length 4 and specifies the delays between each of the
    transfer functions, i.e. delay[0] is the delay between the first and second
    transfer functions.

    """

    # TODO: Make use of previously defined functions to build this one

    timepoints = range(samples + sum(delays))

    # Define source node as pure sine wave
    sine_source = np.array([np.sin(period * t*2*np.pi) for t in timepoints])

    # Calculate response of first transfer function on pure sine signal
    TF_1 = control.matlab.tf([2], [3, 1])
    P1_response = control.matlab.lsim(TF_1, sine_source, timepoints)
    np.random.seed(45)
    P1_response_random_add = (np.random.rand(len(timepoints)) - 0.5) * noiseamp
    TF_1_output_firstpass = P1_response[0] + P1_response_random_add

    return None
