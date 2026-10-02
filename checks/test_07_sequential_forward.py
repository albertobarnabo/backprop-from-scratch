"""Step 7: Sequential.forward"""

import numpy as np

from checks.helpers import BATCH, HIDDEN, N_IN, N_OUT, NET_SIZES, check_close, fail, load

TOYS = "On a test network of three toy layers (first: + 1, second: × 2, third: − 3):\n"
NAMES = ["first", "second", "third"]


class Toy:
    """A toy layer: applies a simple operation, and writes down every input it receives."""

    def __init__(self, name, operation, log):
        self.name, self.operation, self.log = name, operation, log

    def forward(self, x):
        self.log.append((self.name, np.array(x, dtype=float)))
        return self.operation(x)


def toy_network():
    log = []
    layers = [Toy("first", lambda v: v + 1, log), Toy("second", lambda v: 2 * v, log), Toy("third", lambda v: v - 3, log)]
    return load("network").Sequential(layers), log


def test_layers_run_in_order():
    """every layer runs once, in list order, each one getting the previous layer's output"""
    net, log = toy_network()
    x = np.array([[1.0, 2.0]])
    out = net.forward(x)

    called = [name for name, _ in log]
    if sorted(called) != sorted(NAMES):
        fail(TOYS + f"forward called {called}, but the network has {NAMES}. Every layer must run, exactly once.")
    if called != NAMES:
        fail(TOYS + f"forward called the layers in this order: {called}.\n"
                    "Data flows through them in list order: the first layer in the list goes first.")

    received = dict(log)
    if not np.allclose(received["second"], x + 1) or not np.allclose(received["third"], 2 * (x + 1)):
        if np.allclose(received["second"], x):
            fail(TOYS + "every layer got the original x. Each layer must get the output of the previous one.")
        fail(TOYS + "a layer didn't get the output of the previous one as its input.")

    check_close(
        "the network output", out, 2 * (x + 1) - 3,
        mistakes=[
            (x, TOYS + "you returned x unchanged. Keep each layer's output: it's the input of the next layer, "
                       "and the last one is what forward returns."),
            (x + 1, TOYS + "you returned the first layer's output. Return the output of the last layer."),
            (2 * (x + 1), TOYS + "you returned the second layer's output. Return the output of the last layer."),
        ],
        sizes="",
    )


def test_real_network():
    """a Linear → Sigmoid → Linear network turns (batch_size, n_in) into (batch_size, n_out)"""
    layers = load("layers")

    def build():
        rng = np.random.default_rng(6)
        return [layers.Linear(N_IN, HIDDEN, rng), layers.Sigmoid(), layers.Linear(HIDDEN, N_OUT, rng)]

    net = load("network").Sequential(build())
    rng = np.random.default_rng(7)
    for _ in range(2):  # twice, to make sure nothing is left over from the first call
        x = rng.standard_normal((BATCH, N_IN))
        expected = x
        for layer in build():
            expected = layer.forward(expected)
        check_close("the network output", net.forward(x), expected, sizes=NET_SIZES)
        if np.shape(net.layers[0].x) != x.shape:
            fail(
                f"After forward, the first Linear has stored an input of shape {np.shape(net.layers[0].x)}, "
                f"not the whole batch {x.shape}.\n"
                "Pass the whole batch through each layer at once: backward needs every example."
            )
