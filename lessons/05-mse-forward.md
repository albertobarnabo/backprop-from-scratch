# Step 5: `MSE.forward`

> **Write:** `MSE.forward` in `backprop/losses.py`
>
> **Notes:** page 1 ($C_0$), page 2 (total cost), page 3 (multiple outputs)

## Theory

The cost is one number that says how wrong the network is. Learning means making it smaller.

### One example

With one output neuron, page 1 of the notes:

$$C_0 = \left(a^{(L)} - y\right)^2$$

With several output neurons (page 3), add up the squared error of each one:

$$C_0 = \sum_j \left(a_j^{(L)} - y_j\right)^2$$

### A batch of examples

Page 2: the total cost is the **average** of the costs of the $n$ examples:

$$C = \frac{1}{n} \sum_{i=0}^{n-1} C_i = \frac{1}{n} \sum_i \sum_j \left(a_{ij} - y_{ij}\right)^2$$

where $a_{ij}$ is output $j$ of example $i$ (`pred[i, j]`), $y_{ij}$ its target, and $n$ = `batch_size`. Note what you divide by: the number of **examples**, not the number of entries. Each example's cost is a sum over its output neurons; only the examples are averaged.

Some books put a $\frac{1}{2}$ in front, to cancel the 2 of the derivative. Not here: this is exactly the cost from the notes.

### Why forward stores pred and y

The gradient of the cost (next step) needs both, and it's computed later: store them in `self.pred` and `self.y`.

## Your task

In `MSE.forward(self, pred, y)`:

1. store `pred` and `y`;
2. compute $C$ and return it, as a single number.

## Shapes

| | Shape |
|---|---|
| `pred` (in), $a^{(L)}$ | `(batch_size, n_out)` |
| `y` (in) | `(batch_size, n_out)` |
| returned $C$ | a single number |

## Hints

<details>
<summary>Hint 1</summary>

`pred - y` and squaring work entry by entry. Then add everything up into one number, and divide by the number of examples, which you can read from the shape of `pred`.

</details>

<details>
<summary>Hint 2: why not np.mean?</summary>

`np.mean` over the whole array divides by `batch_size × n_out`. With one output neuron (like XOR) it makes no difference, but in general it's a different cost from the one the gradient in step 6 is based on.

</details>

## Check

```bash
uv run tour.py
```

Next: [step 6, `MSE.backward`](06-mse-backward.md).
