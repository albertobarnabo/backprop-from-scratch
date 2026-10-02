"""Step 1: Linear.forward"""

import numpy as np

from checks.helpers import BATCH, N_IN, N_OUT, Unchanged, check_close, check_shape, fail, random_linear


def by_hand(layer, x):
    """z_j = Σ_k w_jk x_k + b_j for every example, one number at a time."""
    z = np.zeros((len(x), N_OUT))
    for i in range(len(x)):
        for j in range(N_OUT):
            z[i, j] = sum(layer.W[j, k] * x[i, k] for k in range(N_IN)) + layer.b[j]
    return z


def test_output_shape():
    """forward returns z with shape (batch_size, n_out)"""
    layer, x = random_linear()
    check_shape(
        "z", layer.forward(x), (BATCH, N_OUT),
        why="Each row of x is one example. For each example you want one z per neuron of this layer.",
    )


def test_output_values():
    """z_j = Σ_k w_jk x_k + b_j, for every example in the batch"""
    layer, x = random_linear()
    expected = by_hand(layer, x)
    check_close(
        "z", layer.forward(x), expected,
        mistakes=[
            (expected - layer.b, "Looks like the bias b is missing."),
            (expected - 2 * layer.b, "The bias is subtracted: it's + b."),
            (1 / (1 + np.exp(-expected)), "You applied the sigmoid inside Linear. The activation is a layer of "
                                          "its own (step 3): Linear only computes z."),
        ],
    )


def test_caches_the_input():
    """forward stores its input in self.x, for backward to use later"""
    layer, x = random_linear()
    original = x.copy()
    layer.forward(x)
    if getattr(layer, "x", None) is None:
        fail("self.x is still None after forward. Store the input x: backward will need it.")
    check_close(
        "self.x", layer.x, original,
        mistakes=[(by_hand(layer, x), "You stored the output z in self.x. Store the input: backward needs it.")],
    )


def test_leaves_x_and_the_parameters_alone():
    """forward doesn't change x, W or b"""
    layer, x = random_linear()
    W, b = layer.W.copy(), layer.b.copy()
    untouched = Unchanged(x=x)
    layer.forward(x)
    untouched.check("forward")
    check_close(
        "self.W after forward", layer.W, W,
        mistakes=[(W.T, "forward replaced self.W with its transpose. Transpose it inside the formula instead "
                        "(W.T gives you a transposed copy to compute with): self.W itself must stay "
                        "(n_out, n_in), for backward and for the learning step.")],
    )
    check_close("self.b after forward", layer.b, b)


def test_works_again_with_new_data():
    """called a second time with new data, forward computes and stores the new values"""
    layer, x = random_linear()
    layer.forward(x)
    new_x = np.random.default_rng(10).standard_normal((BATCH, N_IN))
    z = layer.forward(new_x)
    check_close("z on the second call", z, by_hand(layer, new_x),
                mistakes=[(by_hand(layer, x), "You returned the result of the first call again.")])
    check_close("self.x after the second call", layer.x, new_x,
                mistakes=[(x, "self.x still holds the first call's input: store the input on every call.")])
