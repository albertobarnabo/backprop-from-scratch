# Step 10: train it on XOR

> **Nothing to write.** Every piece is done: let's see it learn.

## The problem

XOR outputs 1 when exactly one of its two inputs is 1:

| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Plot the four points: the 1s sit on one diagonal, the 0s on the other. No single straight line separates them, so a single neuron can't learn XOR. A hidden layer can: it bends the space so that a line works.

![The four XOR points on the unit square, the two y = 1 points (orange) on one diagonal and the two y = 0 points (blue) on the other, next to three attempts at splitting them with one straight line, each leaving at least one point circled in red on the wrong side.](../assets/lessons/10-xor-problem.png)

The network: `Linear(2, 4) → Sigmoid → Linear(4, 1) → Sigmoid`, all four examples in one batch.

## Run it

```bash
uv run train_xor.py
```

Open `train_xor.py`: the whole training loop is five lines, and you wrote every function it calls:

```python
pred = model.forward(x)          # 1. forward pass
loss = loss_fn.forward(pred, y)  # 2. cost
grad = loss_fn.backward()        # 3. dC/da at the output
model.backward(grad)             # 4. backward pass, fills dW and db
model.step(LR)                   # 5. gradient descent update
```

You should see the loss fall towards 0, and predictions close to 0, 1, 1, 0.

![Left, the trained 2-4-1 network's output over the input plane: an orange band through the two y = 1 points between two white 0.5 boundary curves, with blue regions around the two y = 0 points; right, the training loss on a log scale dropping from about 0.28 to 0.0002 over 10,000 steps.](../assets/lessons/10-xor-solved.png)

The tour runs exactly this in step 10:

```bash
uv run tour.py
```

## Experiments

Change the hyperparameters at the top of `train_xor.py` and see what happens:

- **`HIDDEN = 1`**: one hidden neuron. Can it still learn XOR? Why not?
- **`HIDDEN = 2`**: the minimum that can work in theory. Try `SEED` 0 to 9: does it always get there?
- **`LR = 0.1`**: a smaller step. How many more steps does it need?
- **`LR = 10`**: a bigger step, and here it learns *faster*. On a problem this small, big steps are fine.
- **`LR = 100`**: too big. The first steps push the weights so far that the sigmoids saturate: their output sticks at 1, where $\sigma'(z) \approx 0$, so the gradient vanishes and learning stops. Watch the loss freeze at 0.5.
- **`STEPS = 100_000`**: how low does the loss go?

## Going further

- **A new activation.** Add a `ReLU` class in `layers.py`: $\text{ReLU}(z) = \max(0, z)$, whose derivative is 1 where $z > 0$ and 0 elsewhere. Write its `forward` and `backward` the same way as `Sigmoid`, swap it in for the hidden Sigmoid, and compare.
- **Check your own gradients.** `checks/helpers.py` has `numerical_grad`: nudge a number, watch the cost move. That's how every check in this course works, and it's how you test any new layer you write.
- **Bigger data.** Nothing in your code is specific to XOR. Try a network with more inputs and outputs on a dataset of your own.
