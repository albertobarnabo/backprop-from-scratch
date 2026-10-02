"""Step 1: Linear.__init__ and Linear.forward"""

import numpy as np

from checks.helpers import BATCH, N_IN, N_OUT, Unchanged, check_close, check_shape, fail, load, random_linear


def by_hand(layer, x):
    """z_j = Σ_k w_jk x_k + b_j for every example, one number at a time."""
    z = np.zeros((len(x), N_OUT))
    for i in range(len(x)):
        for j in range(N_OUT):
            z[i, j] = sum(layer.W[j, k] * x[i, k] for k in range(N_IN)) + layer.b[j]
    return z


def test_parameters_have_the_right_shapes():
    """__init__ creates W with shape (n_out, n_in) and b with shape (n_out,)"""
    layer = load("layers").Linear(N_IN, N_OUT, np.random.default_rng(0))
    for name in ("W", "b"):
        if getattr(layer, name, None) is None:
            fail(f"The layer has no self.{name}. The parameters must be called W and b: the rest of the code looks for them.")
    check_shape("self.W", layer.W, (N_OUT, N_IN),
                why="W[j, k] is w_jk, the weight from input k to neuron j: one row per neuron.")
    check_shape("self.b", layer.b, (N_OUT,), why="One bias per neuron, as a flat vector.",
                hints={(N_OUT, 1): "b has an extra dimension.", (1, N_OUT): "b has an extra dimension."})


def test_weights_come_from_the_rng():
    """two layers built from the same seed are identical, from different seeds they're not"""
    Linear = load("layers").Linear
    first, again, different = (Linear(N_IN, N_OUT, np.random.default_rng(seed)) for seed in (0, 0, 1))
    for name in ("W", "b"):
        if not np.array_equal(getattr(first, name), getattr(again, name)):
            fail(f"Two layers built from the same seed got different values in {name}.\n"
                 "Draw every random number from the rng you're given: np.random.something() uses a global "
                 "generator that ignores the seed.")
    if np.array_equal(first.W, different.W):
        fail("Layers built from different seeds got the same weights. The weights should be random, "
             "drawn from the rng you're given.")


def test_weights_are_not_all_equal():
    """the weights don't all start with the same value"""
    W = np.asarray(load("layers").Linear(N_IN, N_OUT, np.random.default_rng(0)).W)
    if np.all(W == W.flat[0]):
        fail(f"Every weight starts at {W.flat[0]}. Then every neuron of the layer computes the same thing, "
             "gets the same gradient, and they stay identical forever: the layer is as good as one neuron.\n"
             "Start the weights at different, random values.")


def test_works_without_an_rng():
    """Linear(n_in, n_out) also works when no rng is passed"""
    layer = load("layers").Linear(N_IN, N_OUT)
    check_shape("self.W", layer.W, (N_OUT, N_IN), why="With rng=None, make a random generator yourself.")


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
    """called a second time with new data, forward computes the new values"""
    layer, x = random_linear()
    layer.forward(x)
    new_x = np.random.default_rng(10).standard_normal((BATCH, N_IN))
    check_close("z on the second call", layer.forward(new_x), by_hand(layer, new_x),
                mistakes=[(by_hand(layer, x), "You returned the result of the first call again.")])
