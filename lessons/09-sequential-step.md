# Step 9: `Sequential.step`

> **Write:** `Sequential.step` in `backprop/network.py`
>
> **Notes:** page 2 (the gradient vector)

## Theory

### The gradient points uphill

Page 2 of the notes: put every $\frac{\partial C}{\partial w}$ and $\frac{\partial C}{\partial b}$ of the network into one long vector, and you get the gradient:

$$\nabla C = \begin{bmatrix} \frac{\partial C}{\partial w^{(1)}} & \frac{\partial C}{\partial b^{(1)}} & \cdots & \frac{\partial C}{\partial w^{(L)}} & \frac{\partial C}{\partial b^{(L)}} \end{bmatrix}$$

It's the direction in which the cost **grows** fastest. You want the cost to shrink, so you take a small step the other way.

### Gradient descent

For every parameter:

$$W \leftarrow W - \eta \frac{\partial C}{\partial W} \qquad b \leftarrow b - \eta \frac{\partial C}{\partial b}$$

$\eta$ is the **learning rate** `lr`: how big a step you take. Too small and learning crawls; too big and you jump over the valley.

After `backward`, every Linear layer holds its gradients in `dW` and `db`. `step` uses them. Sigmoid has no parameters, so there's nothing to update there.

One full training step is then: forward, cost, backward, step. Repeat it many times and the network learns.

## Your task

Write `Sequential.step`: one step of gradient descent, with learning rate `lr`, for every layer that has parameters.

Questions to ask yourself:

- Which layers have something to learn, and how can you tell them apart?
- After backward, where are the gradients?

## Hints

<details>
<summary>Hint 1: which layers have parameters?</summary>

`hasattr(layer, "W")` tells you whether a layer has a `W`. Sigmoid doesn't.

</details>

## Check

```bash
uv run tour.py
```

That's `network.py` done. Next: [step 10, train it on XOR](10-train-xor.md).
