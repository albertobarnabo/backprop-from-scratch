"""Step 2: Linear.backward"""

import numpy as np

from checks.helpers import (
    BATCH, N_IN, N_OUT, SAME_SHAPE, Unchanged, check_close, check_shape, fail, numerical_grad, random_linear,
)

AVERAGED = (
    "The values are exactly 1/batch_size of the right ones: you averaged over the examples.\n"
    "Sum over them instead. The 1/n of the average is already inside grad_out: "
    "the loss puts it there (step 6)."
)

DB_SHAPES = {
    (BATCH,): "db has one value per example. You want one per neuron: add up over the examples, the batch axis.",
    (): "db is a single number. Add up over the examples only, so that each neuron keeps its own total.",
    (1, N_OUT): "db has an extra dimension: it should be a flat vector, like b.",
}


def make_case(seed=1):
    layer, x = random_linear()
    grad_out = np.random.default_rng(seed).standard_normal((BATCH, N_OUT))

    def cost():
        # A stand-in for whatever sits above this layer: this C has dC/dz = grad_out exactly.
        return np.sum(layer.forward(x) * grad_out)

    return layer, x, grad_out, cost


def test_returns_dx_shaped_like_x():
    """backward returns dC/dx with the shape of x: (batch_size, n_in)"""
    layer, x, grad_out, _ = make_case()
    layer.forward(x)
    check_shape("the returned dC/dx", layer.backward(grad_out), (BATCH, N_IN), why=SAME_SHAPE)


def test_stores_dW_and_db_shaped_like_W_and_b():
    """self.dW has the shape of W (n_out, n_in), self.db the shape of b (n_out,)"""
    layer, x, grad_out, _ = make_case()
    layer.forward(x)
    layer.backward(grad_out)
    check_shape("self.dW", layer.dW, (N_OUT, N_IN), why=SAME_SHAPE)
    check_shape("self.db", layer.db, (N_OUT,), why=SAME_SHAPE, hints=DB_SHAPES)


def test_dx_values():
    """the returned dC/dx matches the gradient measured numerically"""
    layer, x, grad_out, cost = make_case()
    expected = numerical_grad(cost, x)
    layer.forward(x)
    check_close("the returned dC/dx", layer.backward(grad_out), expected, why=SAME_SHAPE)


def test_dW_values():
    """self.dW matches the gradient measured numerically"""
    layer, x, grad_out, cost = make_case()
    expected = numerical_grad(cost, layer.W)
    layer.forward(x)
    layer.backward(grad_out)
    check_close("self.dW", layer.dW, expected, mistakes=[(expected / BATCH, AVERAGED)], why=SAME_SHAPE)


def test_db_values():
    """self.db matches the gradient measured numerically"""
    layer, x, grad_out, cost = make_case()
    expected = numerical_grad(cost, layer.b)
    layer.forward(x)
    layer.backward(grad_out)
    check_close("self.db", layer.db, expected, mistakes=[(expected / BATCH, AVERAGED)],
                why=SAME_SHAPE, hints=DB_SHAPES)


def test_leaves_the_parameters_alone():
    """backward only computes gradients: W, b, x and grad_out stay as they were"""
    layer, x, grad_out, _ = make_case()
    layer.forward(x)
    before = {"W": layer.W.copy(), "b": layer.b.copy()}
    untouched = Unchanged(x=x, grad_out=grad_out)
    layer.backward(grad_out)
    untouched.check("backward")
    for name, value in before.items():
        now = getattr(layer, name)
        if np.shape(now) != value.shape or not np.allclose(now, value):
            fail(f"backward changed self.{name}. It only computes gradients: "
                 "updating the parameters is the learning step's job (step 9).")


def test_works_again_with_new_data():
    """called a second time, backward overwrites dW and db with the new gradients"""
    layer, x, grad_out, cost = make_case(seed=1)
    first_dW = numerical_grad(cost, layer.W)
    first_db = numerical_grad(cost, layer.b)
    layer.forward(x)
    layer.backward(grad_out)

    new_grad_out = np.random.default_rng(11).standard_normal((BATCH, N_OUT))

    def new_cost():
        return np.sum(layer.forward(x) * new_grad_out)

    expected_dW = numerical_grad(new_cost, layer.W)
    expected_db = numerical_grad(new_cost, layer.b)
    layer.forward(x)
    layer.backward(new_grad_out)

    piled_up = ("self.{} kept the previous call's gradient and added the new one to it (+=). Overwrite it "
                "instead: each backward computes the gradient from scratch.")
    check_close("self.dW on the second call", layer.dW, expected_dW,
                mistakes=[(first_dW + expected_dW, piled_up.format("dW")),
                          (first_dW, "self.dW still holds the first call's gradient.")])
    check_close("self.db on the second call", layer.db, expected_db,
                mistakes=[(first_db + expected_db, piled_up.format("db")),
                          (first_db, "self.db still holds the first call's gradient.")])
