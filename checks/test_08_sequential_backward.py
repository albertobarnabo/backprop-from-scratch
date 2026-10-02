"""Step 8: Sequential.backward"""

import numpy as np

from checks.helpers import (
    BATCH, HIDDEN, N_IN, N_OUT, NET_SIZES, SAME_SHAPE, check_close, fail, load, numerical_grad,
)

TOYS = "On a test network of three toy layers whose backward multiplies the gradient by 2, 3 and 5:\n"
NAMES = ["first", "second", "third"]


class Scale:
    """A toy layer: multiplies by a constant, and writes its name in a log when backward runs."""

    def __init__(self, name, factor, log, forward_log=None):
        self.name, self.factor, self.log = name, factor, log
        self.forward_log = forward_log if forward_log is not None else []

    def forward(self, x):
        self.forward_log.append(self.name)
        return self.factor * x

    def backward(self, grad_out):
        self.log.append(self.name)
        return self.factor * grad_out


def toy_network(forward_log=None):
    log = []
    layers = [Scale(name, factor, log, forward_log) for name, factor in zip(NAMES, [2, 3, 5])]
    return load("network").Sequential(layers), log


def test_layers_run_in_reverse():
    """backward visits every layer once, from the last one to the first one"""
    net, log = toy_network()
    net.forward(np.ones((1, 2)))
    net.backward(np.ones((1, 2)))
    if sorted(log) != sorted(NAMES):
        if log == ["third", "second"]:
            why = ("The first layer's backward never ran. A range's stop value is excluded: "
                   "range(len(layers) - 1, 0, -1) stops before index 0.")
        else:
            why = ("Every layer's backward must run exactly once, even for layers without parameters like Sigmoid: "
                   "their factor is part of the chain.")
        fail(TOYS + f"backward called {log}, but the network has {NAMES}.\n" + why)
    if log != NAMES[::-1]:
        fail(
            TOYS + f"backward called the layers in this order: {log}.\n"
            "Backpropagation walks the network in reverse: the last layer gets the gradient first, "
            "because it sits right below the loss."
        )


def test_gradient_flows_through_every_layer():
    """each layer's backward receives what the layer above it returned"""
    grad = np.array([[0.5, -1.5]])
    net, _ = toy_network()
    net.forward(np.ones((1, 2)))
    check_close(
        "the returned gradient", net.backward(grad), grad * 30,
        mistakes=[
            (grad * 2, TOYS + "every layer received the original grad. Pass it along: what one layer's backward "
                              "returns is what the next backward call receives."),
            (grad * 5, TOYS + "only the last layer ran its backward, or only its result came back. Return the "
                              "gradient after it has gone through every layer."),
            (grad, TOYS + "you returned grad unchanged. Keep what each layer's backward returns: "
                          "it's what the next backward call receives."),
            (np.ones_like(grad) * 30, TOYS + "the grad you received was ignored: start from it, "
                                             "it's how the cost reacts to the network's output."),
        ],
        sizes="",
    )


def test_leaves_the_network_alone():
    """backward doesn't reorder the layers, and works every time it's called"""
    forward_log = []
    net, log = toy_network(forward_log)
    for call in ("first", "second"):
        forward_log.clear()
        log.clear()
        net.forward(np.ones((1, 2)))
        if forward_log != NAMES:
            fail(
                TOYS + f"after one backward, forward ran the layers in this order: {forward_log}.\n"
                "backward changed the order of the layers, e.g. with list.reverse() or by storing the list "
                "reversed: that flips the network itself. Walk the list backwards without changing it."
            )
        net.backward(np.ones((1, 2)))
        if log != NAMES[::-1]:
            fail(
                TOYS + f"on the {call} call, backward called {log} instead of {NAMES[::-1]}.\n"
                "backward must work every time, not just once: an iterator like reversed(...) kept from an "
                "earlier call can only be walked through once."
            )


def test_full_network_gradients():
    """dW and db of every Linear, and the returned dC/dx, match numerical gradients through the MSE"""
    layers, losses = load("layers"), load("losses")
    rng = np.random.default_rng(8)
    first, second = layers.Linear(N_IN, HIDDEN, rng), layers.Linear(HIDDEN, N_OUT, rng)
    first.b = rng.standard_normal(HIDDEN)
    second.b = rng.standard_normal(N_OUT)
    net = load("network").Sequential([first, layers.Sigmoid(), second, layers.Sigmoid()])
    loss = losses.MSE()
    x = rng.standard_normal((BATCH, N_IN))
    y = rng.integers(0, 2, (BATCH, N_OUT)).astype(float)

    def cost():
        return loss.forward(net.forward(x), y)

    expected = {
        "first Linear's dW": numerical_grad(cost, first.W),
        "first Linear's db": numerical_grad(cost, first.b),
        "second Linear's dW": numerical_grad(cost, second.W),
        "second Linear's db": numerical_grad(cost, second.b),
        "the returned dC/dx": numerical_grad(cost, x),
    }

    cost()
    dx = net.backward(loss.backward())
    got = {
        "first Linear's dW": first.dW,
        "first Linear's db": first.db,
        "second Linear's dW": second.dW,
        "second Linear's db": second.db,
        "the returned dC/dx": dx,
    }
    for name in expected:
        check_close(name, got[name], expected[name], why=SAME_SHAPE, sizes=NET_SIZES)
