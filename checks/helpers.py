"""Shared tools for the checks. You never need to edit this file."""

import importlib
import os

import numpy as np

# The package being checked: yours by default. BACKPROP_IMPL=mlp checks a reference solution instead.
IMPL = os.environ.get("BACKPROP_IMPL", "backprop")

# Every check uses these sizes. They are all different on purpose:
# if an error says (4, 5), you can tell it means (n_out, batch_size).
BATCH, N_IN, N_OUT = 5, 3, 4
HIDDEN = 6  # the hidden layer of the test networks in steps 7 to 9
SIZES = f"[batch_size={BATCH}, n_in={N_IN}, n_out={N_OUT}]"
NET_SIZES = f"[batch_size={BATCH}, n_in={N_IN}, hidden={HIDDEN}, n_out={N_OUT}]"

SAME_SHAPE = (
    "A gradient always has the shape of the thing it is the gradient of: "
    "dC/dW is shaped like W, dC/db like b, dC/dx like x."
)

# Every value off by the same factor usually means one specific slip.
FACTORS = [
    (-1, "The sign is flipped. Gradients only describe how C changes: the minus sign of gradient descent "
         "belongs in step 9, nowhere else."),
    (1 / BATCH, "You divided by batch_size. The only 1/n in the whole network lives in MSE (steps 5 and 6): "
                "everywhere else, examples are summed or kept apart, never averaged."),
    (BATCH, "You multiplied by batch_size, or a 1/batch_size is missing."),
    (1 / N_OUT, "You divided by n_out, the number of neurons. Nothing here averages over neurons."),
    (BATCH / N_OUT, "You divided by n_out where it should be batch_size: the number of examples is the "
                    "first dim, shape[0], not shape[1]."),
    (2, "Every value is twice the right one: a factor of 2 too many."),
    (1 / 2, "Every value is half the right one: a factor of 2 is missing."),
]


def load(module):
    """Imports one of your files: load("layers"), load("losses"), load("network")."""
    return importlib.import_module(f"{IMPL}.{module}")


def fail(message):
    raise AssertionError(message)


def show(array):
    return np.array2string(np.asarray(array, dtype=float), precision=4, suppress_small=True)


def as_array(name, got, number=False):
    """Turns what you returned into an array, or fails with a message saying what it is instead."""
    if got is None:
        where = "store it on self" if name.startswith("self.") else "return it"
        fail(f"{name} is None. Did you forget to {where}?")
    if isinstance(got, tuple):
        fail(
            f"{name} is a tuple of {len(got)} values. Return only one thing: "
            "values like dW and db are stored on self, not returned."
        )
    if number:
        if np.ndim(got) != 0 or not np.issubdtype(np.asarray(got).dtype, np.number):
            fail(f"{name} should be a single number, but it has shape {np.shape(got)}.")
        return np.asarray(got, dtype=float)
    if not isinstance(got, np.ndarray):
        fail(f"{name} is a {type(got).__name__}, not a numpy array. Keep everything as numpy arrays.")
    return got


def check_shape(name, got, shape, why="", sizes=SIZES, hints=None):
    """Fails with a readable message if `got` is missing or doesn't have the expected shape.

    `hints` maps a wrong shape to an explanation of what probably produced it.
    """
    got = as_array(name, got, number=shape == ())
    if got.shape == shape:
        return got

    message = f"{name} has shape {got.shape}, but it should be {shape}." + (f"  {sizes}" if sizes else "")
    if hints and got.shape in hints:
        message += "\n" + hints[got.shape]
    elif len(shape) == 2 and got.shape == shape[::-1]:
        message += "\nIt looks transposed."
    if why:
        message += "\n" + why
    fail(message)


def check_close(name, got, expected, mistakes=(), why="", sizes=SIZES, hints=None):
    """Fails if `got` doesn't match `expected`.

    `mistakes` lists (wrong_value, explanation) pairs: if `got` matches one of them,
    the explanation is shown instead of a generic message.
    """
    expected = np.asarray(expected, dtype=float)
    got = check_shape(name, got, expected.shape, why, sizes, hints).astype(float)
    if np.allclose(got, expected, rtol=1e-5, atol=1e-7):
        return

    header = f"{name} has the right shape but the wrong values."
    for wrong, explanation in mistakes:
        if np.allclose(got, wrong, rtol=1e-5, atol=1e-7):
            fail(f"{header}\n{explanation}")

    # Every value off by the same factor?
    if np.any(expected != 0):
        k = float(np.sum(got * expected) / np.sum(expected * expected))
        if np.allclose(got, k * expected, rtol=1e-5, atol=1e-7):
            for factor, explanation in FACTORS:
                if np.isclose(k, factor):
                    fail(f"{header}\n{explanation}")
            fail(f"{header}\nEvery value is exactly {k:.4g} × the right one.")

    fail(f"{header}\nexpected:\n{show(expected)}\ngot:\n{show(got)}")


class Unchanged:
    """Remembers arrays, to check later that nobody changed them in place.

    `b = a` doesn't copy: both names point to the same array, so `b -= y` also changes `a`.
    """

    def __init__(self, **arrays):
        self.arrays = arrays
        self.copies = {name: np.array(array, copy=True) for name, array in arrays.items()}

    def check(self, during):
        for name, array in self.arrays.items():
            if not np.array_equal(array, self.copies[name]):
                fail(
                    f"{during} changed {name} in place.\n"
                    "An in-place operation (+=, -=, *=, or writing into a slice) on an array changes it everywhere, "
                    "and `a = b` doesn't copy: both names point to the same array.\n"
                    "Write the result into a new array instead, e.g. `c = a - b` rather than `a -= b`."
                )


def numerical_grad(cost, x, eps=1e-6):
    """dC/dx measured without any calculus: nudge each entry of x up and down by eps
    and see how much the cost moves. `cost()` must recompute C from the current x.

    x is changed in place during the measurement and restored afterwards.
    """
    grad = np.zeros(x.shape)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps
        up = float(cost())
        x[idx] = old - eps
        down = float(cost())
        x[idx] = old
        grad[idx] = (up - down) / (2 * eps)
    return grad


def random_linear(seed=0):
    """A Linear(n_in, n_out) layer with a non-zero bias, and a batch of inputs for it."""
    rng = np.random.default_rng(seed)
    layer = load("layers").Linear(N_IN, N_OUT, rng)
    layer.b = rng.standard_normal(N_OUT)  # non-zero, so a forgotten bias shows up
    x = rng.standard_normal((BATCH, N_IN))
    return layer, x
