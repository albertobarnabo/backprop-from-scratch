"""Step 6: MSE.backward"""

import numpy as np

from checks.helpers import (
    BATCH, N_OUT, SAME_SHAPE, Unchanged, check_close, check_shape, load, numerical_grad,
)
from checks.test_05_mse_forward import by_hand


def make_case(seed=5):
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0, 1, (BATCH, N_OUT))
    y = rng.integers(0, 2, (BATCH, N_OUT)).astype(float)
    return load("losses").MSE(), pred, y


def test_returns_grad_shaped_like_pred():
    """backward returns dC/da with the shape of pred: (batch_size, n_out)"""
    loss, pred, y = make_case()
    loss.forward(pred, y)
    check_shape("the returned dC/da", loss.backward(), (BATCH, N_OUT), why=SAME_SHAPE)


def test_values():
    """the returned dC/da matches the gradient measured numerically"""
    loss, pred, y = make_case()
    expected = numerical_grad(lambda: by_hand(pred, y), pred)
    loss.forward(pred, y)
    check_close(
        "the returned dC/da", loss.backward(), expected,
        mistakes=[
            (expected / 2, "The 2 is missing: the derivative of (a − y)² is 2(a − y)."),
            (expected * BATCH,
             "The 1/batch_size is missing. C is an average over the examples, so its gradient keeps the 1/n."),
            (-expected, "The sign is flipped: it's prediction minus target, (a − y)."),
            (expected / N_OUT,
             "You divided by batch_size × n_out. Only the examples are averaged, so only divide by batch_size."),
            (expected * BATCH / N_OUT, "You divided by n_out, the number of output neurons (shape[1]). "
                                       "The number of examples is the first dim: shape[0]."),
            (expected / BATCH, "You divided by batch_size twice."),
            (expected * pred * (1 - pred),
             "You multiplied by σ'(z). The loss only knows about its input a: σ'(z) is the Sigmoid layer's "
             "factor, and its backward applies it (step 4)."),
        ],
    )


def test_leaves_pred_and_y_alone():
    """backward doesn't change the stored pred and y (they are the network's own arrays)"""
    loss, pred, y = make_case()
    loss.forward(pred, y)
    untouched = Unchanged(pred=pred, y=y)
    loss.backward()
    untouched.check("backward")


def test_works_again_with_new_data():
    """called a second time, backward uses the values of the latest forward"""
    loss, pred, y = make_case()
    first = numerical_grad(lambda: by_hand(pred, y), pred)
    loss.forward(pred, y)
    loss.backward()
    _, new_pred, new_y = make_case(seed=15)
    expected = numerical_grad(lambda: by_hand(new_pred, new_y), new_pred)
    loss.forward(new_pred, new_y)
    check_close("the returned dC/da on the second call", loss.backward(), expected,
                mistakes=[(first, "You returned the first call's gradient again: compute it from the latest "
                                  "self.pred and self.y every time.")])
