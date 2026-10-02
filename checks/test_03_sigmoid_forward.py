"""Step 3: Sigmoid.forward"""

import math

import numpy as np

from checks.helpers import BATCH, N_OUT, Unchanged, check_close, check_shape, fail, load


def random_z(seed=2):
    return np.random.default_rng(seed).uniform(-6, 6, (BATCH, N_OUT))


def by_hand(z):
    return np.vectorize(lambda v: 1 / (1 + math.exp(-v)))(z)


def test_output_shape():
    """forward returns σ(z) with the same shape as z"""
    z = random_z()
    check_shape(
        "σ(z)", load("layers").Sigmoid().forward(z), z.shape,
        why="Sigmoid works on every number on its own: the shape never changes.",
    )


def test_output_values():
    """σ(z) = 1 / (1 + e^(−z)), for every entry"""
    z = random_z()
    expected = by_hand(z)
    check_close(
        "σ(z)", load("layers").Sigmoid().forward(z), expected,
        mistakes=[
            (1 - expected, "You computed 1 − σ(z), which is σ(−z). Check the sign in the exponent, e^(−z), "
                           "and the numerator, 1."),
            (z, "You returned z unchanged: apply σ to it."),
            (1 + np.exp(-z), "Operator precedence: 1/1 + np.exp(-z) means (1/1) + e^(−z). "
                             "Put the whole denominator in parentheses."),
        ],
    )


def test_caches_the_output():
    """forward stores σ(z) in self.sig_z, for backward to use later"""
    sigmoid = load("layers").Sigmoid()
    z = random_z()
    s = by_hand(z)
    sigmoid.forward(z)
    if getattr(sigmoid, "sig_z", None) is None:
        fail("self.sig_z is still None after forward. Store σ(z): backward will need it.")
    check_close(
        "self.sig_z", sigmoid.sig_z, s,
        mistakes=[
            (random_z(), "You stored the input z in self.sig_z. Store the output σ(z): backward needs it."),
            (s * (1 - s), "You stored σ'(z) in self.sig_z. Store σ(z) itself: backward computes σ'(z) from it."),
        ],
    )


def test_leaves_z_alone():
    """forward doesn't change its input z"""
    z = random_z()
    untouched = Unchanged(z=z)
    load("layers").Sigmoid().forward(z)
    untouched.check("forward")


def test_works_again_with_new_data():
    """called a second time with new data, forward computes and stores the new values"""
    sigmoid = load("layers").Sigmoid()
    z = random_z()
    sigmoid.forward(z)
    new_z = random_z(seed=12)
    check_close("σ(z) on the second call", sigmoid.forward(new_z), by_hand(new_z),
                mistakes=[(by_hand(z), "You returned the result of the first call again.")])
    check_close("self.sig_z after the second call", sigmoid.sig_z, by_hand(new_z),
                mistakes=[(by_hand(z), "self.sig_z still holds the first call's value: store it on every call.")])
