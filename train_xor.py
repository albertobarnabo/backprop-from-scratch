"""The payoff: your network learns XOR.  Run it with:  uv run train_xor.py

It works once every step of `uv run tour.py` is done.
"""

import numpy as np

from backprop.layers import Linear, Sigmoid
from backprop.losses import MSE
from backprop.network import Sequential

# --- Data: XOR, all four examples in one batch ---
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)  # (4, 2)
y = np.array([[0], [1], [1], [0]], dtype=float)              # (4, 1)

# --- Hyperparameters ---
SEED = 0
HIDDEN = 4
LR = 1.0
STEPS = 10_000
PRINT_EVERY = 1_000

# --- Model and loss ---
rng = np.random.default_rng(SEED)
model = Sequential([
    Linear(2, HIDDEN, rng),
    Sigmoid(),
    Linear(HIDDEN, 1, rng),
    Sigmoid(),
])
loss_fn = MSE()

# --- Training loop ---
for step in range(STEPS + 1):
    pred = model.forward(x)          # 1. forward pass
    loss = loss_fn.forward(pred, y)  # 2. cost
    grad = loss_fn.backward()        # 3. dC/da at the output
    model.backward(grad)             # 4. backward pass, fills dW and db
    model.step(LR)                   # 5. gradient descent update

    if step % PRINT_EVERY == 0:
        print(f"step {step:>6}  loss {loss:.6f}")

# --- Result ---
pred = model.forward(x)
print("\ninput    target  prediction")
for xi, yi, pi in zip(x, y, pred):
    print(f"{xi}  {yi[0]:.0f}       {pi[0]:.4f}")
