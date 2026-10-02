"""Step 10: train the whole thing on XOR"""

import numpy as np

from checks.helpers import fail, load


def test_learns_xor():
    """Linear(2, 4) → Sigmoid → Linear(4, 1) → Sigmoid learns XOR in 10,000 steps"""
    layers, losses, network = load("layers"), load("losses"), load("network")
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    rng = np.random.default_rng(0)
    model = network.Sequential([layers.Linear(2, 4, rng), layers.Sigmoid(), layers.Linear(4, 1, rng), layers.Sigmoid()])
    loss_fn = losses.MSE()

    history = []
    for step in range(10_001):
        pred = model.forward(x)
        loss = loss_fn.forward(pred, y)
        model.backward(loss_fn.backward())
        model.step(1.0)
        if step % 2_000 == 0:
            history.append(f"step {step:>6}  loss {float(loss):.6f}")

    pred = model.forward(x)
    if not float(loss) < 0.01 or not np.array_equal(np.round(pred), y):
        fail(
            "Every piece passed its own check, but the network didn't learn XOR.\n"
            + "\n".join(history)
            + f"\npredictions: {np.round(pred.ravel(), 4)}, targets: {y.ravel()}\n"
            "Look for something that survives from one training step to the next: "
            "a value kept on self and reused, or an array changed in place."
        )
