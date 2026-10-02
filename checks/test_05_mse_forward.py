"""Step 5: MSE.forward"""

import numpy as np

from checks.helpers import BATCH, N_OUT, Unchanged, check_close, check_shape, fail, load


def make_case(seed=4):
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0, 1, (BATCH, N_OUT))
    y = rng.integers(0, 2, (BATCH, N_OUT)).astype(float)
    return pred, y


def by_hand(pred, y):
    """C = 1/n Σ_i Σ_j (a_ij − y_ij)², one number at a time."""
    total = 0.0
    for i in range(BATCH):
        total += sum((pred[i, j] - y[i, j]) ** 2 for j in range(N_OUT))
    return total / BATCH


def test_returns_a_single_number():
    """forward returns the cost C as a single number"""
    pred, y = make_case()
    check_shape(
        "C", load("losses").MSE().forward(pred, y), (),
        why="Add up the squared errors of every output neuron and every example into one number.",
    )


def test_value():
    """C = 1/n Σ_examples Σ_j (a_j − y_j)², with n = batch_size"""
    pred, y = make_case()
    expected = by_hand(pred, y)
    total = expected * BATCH

    check_close(
        "C", load("losses").MSE().forward(pred, y), expected,
        mistakes=[
            (total, "You added up every example but didn't average: divide by the number of examples."),
            (total / (BATCH * N_OUT),
             "You divided by batch_size × n_out (that's what a mean over every entry does).\n"
             "Divide by the number of examples only: the cost of one example is the sum over its "
             "output neurons, C₀ = Σ_j (a_j − y_j)², and C is the average of those."),
            (total / N_OUT, "You divided by n_out, the number of output neurons (shape[1]). "
                            "The number of examples is the first dim: shape[0]."),
            (total / (2 * BATCH), "That's the ½ convention some books use. Here C has no ½: "
                                  "C = 1/n Σ Σ (a − y)²."),
            (np.sqrt(expected), "That's the square root of C: no root in the mean squared error."),
            (np.sum(pred - y) / BATCH, "The square is missing: positive and negative errors cancel out."),
            (np.sqrt(total) / BATCH, "Square the errors and add them up: there's no square root anywhere."),
        ],
    )


def test_leaves_pred_and_y_alone():
    """forward doesn't change pred or y"""
    pred, y = make_case()
    untouched = Unchanged(pred=pred, y=y)
    load("losses").MSE().forward(pred, y)
    untouched.check("forward")


def test_caches_pred_and_y():
    """forward stores pred and y in self.pred and self.y, for backward to use later"""
    loss = load("losses").MSE()
    pred, y = make_case()
    loss.forward(pred.copy(), y.copy())
    if getattr(loss, "pred", None) is None or getattr(loss, "y", None) is None:
        fail("self.pred or self.y is still None after forward. Store both: backward will need them.")
    check_close("self.pred", loss.pred, pred, mistakes=[(y, "You swapped them: self.pred got y.")])
    check_close("self.y", loss.y, y)


def test_works_again_with_new_data():
    """called a second time with new data, forward computes and stores the new values"""
    loss = load("losses").MSE()
    pred, y = make_case()
    loss.forward(pred, y)
    new_pred, new_y = make_case(seed=14)
    check_close("C on the second call", loss.forward(new_pred, new_y), by_hand(new_pred, new_y),
                mistakes=[(by_hand(pred, y), "You returned the result of the first call again.")])
    check_close("self.pred after the second call", loss.pred, new_pred,
                mistakes=[(pred, "self.pred still holds the first call's value: store it on every call.")])
