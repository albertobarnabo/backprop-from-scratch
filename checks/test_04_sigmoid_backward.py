"""Step 4: Sigmoid.backward"""

import numpy as np

from checks.helpers import BATCH, N_OUT, Unchanged, check_close, check_shape, load, numerical_grad

ENTRY_BY_ENTRY = ("Sigmoid works entry by entry: σ'(z) and grad_out have the same shape, and you multiply them "
                  "element-wise with *. No @, no np.diag.")
SQUARE = ("A square result means a matrix product (@) mixed different entries together. "
          "Sigmoid works entry by entry: multiply element-wise with *.")


def make_case(seed=3):
    sigmoid = load("layers").Sigmoid()
    rng = np.random.default_rng(seed)
    z = rng.uniform(-4, 4, (BATCH, N_OUT))
    grad_out = rng.standard_normal((BATCH, N_OUT))

    def cost():
        # A stand-in for whatever sits above this layer: this C has dC/da = grad_out exactly.
        return np.sum(sigmoid.forward(z) * grad_out)

    return sigmoid, z, grad_out, cost


def test_returns_dz_shaped_like_z():
    """backward returns dC/dz with the same shape as grad_out"""
    sigmoid, z, grad_out, _ = make_case()
    sigmoid.forward(z)
    check_shape("the returned dC/dz", sigmoid.backward(grad_out), z.shape, why=ENTRY_BY_ENTRY,
                hints={(BATCH, BATCH): SQUARE, (N_OUT, N_OUT): SQUARE})


def test_dz_values():
    """the returned dC/dz matches the gradient measured numerically"""
    sigmoid, z, grad_out, cost = make_case()
    expected = numerical_grad(cost, z)

    s = 1 / (1 + np.exp(-z))
    s_of_s = 1 / (1 + np.exp(-s))
    sigmoid.forward(z)
    check_close(
        "the returned dC/dz", sigmoid.backward(grad_out), expected,
        mistakes=[
            (s * (1 - s), "You returned σ'(z) alone. Chain rule: multiply it by grad_out, the dC/da coming from above."),
            (grad_out, "You returned grad_out unchanged. Multiply it by σ'(z), this layer's own factor."),
            (grad_out * s_of_s * (1 - s_of_s),
             "Looks like you applied σ twice: σ(σ(z)). If you kept σ(z) from forward, it already is σ(z): "
             "use it as it is."),
            (grad_out * (1 - s ** 2), "1 − σ² looks like the derivative of tanh, 1 − tanh². For the sigmoid it's σ(z)(1 − σ(z))."),
            (grad_out * s, "Half of σ'(z) is missing: it's σ(z) × (1 − σ(z))."),
            (grad_out * (1 - s), "Half of σ'(z) is missing: it's σ(z) × (1 − σ(z))."),
        ],
    )


def test_leaves_its_inputs_alone():
    """backward doesn't change grad_out, nor the σ(z) that forward returned"""
    sigmoid, z, grad_out, _ = make_case()
    out = sigmoid.forward(z)
    untouched = Unchanged(grad_out=grad_out, **{"the σ(z) returned by forward": out})
    sigmoid.backward(grad_out)
    untouched.check("backward")


def test_works_again_with_new_data():
    """called a second time, backward uses the values of the latest forward"""
    sigmoid, z, grad_out, _ = make_case(seed=3)
    sigmoid.forward(z)
    sigmoid.backward(grad_out)

    _, new_z, new_grad_out, new_cost = make_case(seed=13)
    expected = numerical_grad(new_cost, new_z)
    sigmoid.forward(new_z)
    check_close("the returned dC/dz on the second call", sigmoid.backward(new_grad_out), expected)
