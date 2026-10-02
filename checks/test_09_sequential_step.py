"""Step 9: Sequential.step"""

import numpy as np

from checks.helpers import HIDDEN, N_IN, N_OUT, NET_SIZES, check_close, load

LR = 0.37  # an unusual value, so a hard-coded learning rate shows up


def test_updates_every_linear_layer():
    """W ← W − lr·dW and b ← b − lr·db for every Linear, skipping layers without parameters"""
    layers = load("layers")
    rng = np.random.default_rng(9)
    linears = [layers.Linear(N_IN, HIDDEN, rng), layers.Linear(HIDDEN, N_OUT, rng)]
    net = load("network").Sequential([linears[0], layers.Sigmoid(), linears[1]])

    before = []
    for layer in linears:
        layer.dW = rng.standard_normal(layer.W.shape)
        layer.db = rng.standard_normal(layer.b.shape)
        before.append({"W": layer.W.copy(), "b": layer.b.copy(), "dW": layer.dW.copy(), "db": layer.db.copy()})

    net.step(LR)

    for name, layer, old in zip(["first", "second"], linears, before):
        for p in ["W", "b"]:
            value, grad = old[p], old["d" + p]
            check_close(
                f"the {name} Linear's {p} after step", getattr(layer, p), value - LR * grad,
                mistakes=[
                    (value, f"{p} didn't change. Assign the new value to layer.{p} itself: "
                            "changing a local variable doesn't change the layer."),
                    (value + LR * grad, f"You moved {p} up the gradient. The gradient points uphill: subtract it."),
                    (value - grad, f"The step isn't scaled: multiply d{p} by the learning rate lr."),
                    (value - 0.1 * grad, "The learning rate looks hard-coded to 0.1: use the lr argument."),
                    (value - 0.01 * grad, "The learning rate looks hard-coded to 0.01: use the lr argument."),
                    (value - LR * LR * grad, f"lr was applied twice: {p} − lr·d{p} needs it once."),
                ],
                sizes=NET_SIZES,
            )
