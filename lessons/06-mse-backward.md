# Step 6: `MSE.backward`

> **Write:** `MSE.backward` in `backprop/losses.py`
>
> **Notes:** page 1, where $\frac{\partial C_0}{\partial a^{(L)}} = 2(a^{(L)} - y)$

## Theory

This is where backpropagation **starts**. The cost sits at the top of the chain, so there's nothing above it and no `grad_out` argument. This method creates the first gradient, the one that then flows down through the whole network.

### One example

From page 1 of the notes:

$$\frac{\partial C_0}{\partial a^{(L)}} = 2\left(a^{(L)} - y\right)$$

### A batch of examples

$$C = \frac{1}{n} \sum_i \sum_j \left(a_{ij} - y_{ij}\right)^2$$

Take the derivative with respect to one entry $a_{ij}$. Only one term of the double sum contains it, so all the others vanish:

$$\frac{\partial C}{\partial a_{ij}} = \frac{2}{n} \left(a_{ij} - y_{ij}\right)$$

### The 1/n, again

That $\frac{1}{n}$ is the one from step 2. Because the loss puts it into the very first gradient, it travels down through every layer, and `Linear.backward` only has to **sum** over the examples to end up with the average. Put it here, and only here.

## Your task

Write `MSE.backward`, which returns $\frac{\partial C}{\partial a}$ for every entry.

Questions to ask yourself:

- backward receives nothing: did forward keep what the formula needs?

## Shapes

| | Shape |
|---|---|
| `pred`, from forward | `(batch_size, n_out)` |
| `y`, from forward | `(batch_size, n_out)` |
| returned $\frac{\partial C}{\partial a}$ | `(batch_size, n_out)`, like `pred` |

## Hints

<details>
<summary>Hint 1</summary>

The result is one number per entry of `pred`, so it's all element-wise: no sum this time. $n$ is the number of examples, the first dim of `pred`.

</details>

## Check

```bash
uv run tour.py
```

That's `losses.py` done. Next: [step 7, `Sequential.forward`](07-sequential-forward.md).
