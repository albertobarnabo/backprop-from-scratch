# Step 1: `Linear.forward`

> **Write:** `Linear.forward` in `backprop/layers.py`
>
> **Notes:** page 1 ($z^{(L)}$) and page 3 ($w_{jk}$)

## Theory

### One neuron

A neuron takes the activation of the previous layer, scales it by a weight, and adds a bias:

$$z^{(L)} = w^{(L)} a^{(L-1)} + b^{(L)}$$

### A layer of neurons

With several neurons on both sides, every neuron $j$ of this layer is connected to every neuron $k$ of the previous one, each connection with its own weight $w_{jk}$:

$$z_j = w_{j0} a_0 + w_{j1} a_1 + w_{j2} a_2 + b_j = \sum_k w_{jk} a_k + b_j$$

Put all the weights in a matrix `W`, with `W[j, k]` $= w_{jk}$. Row $j$ holds all the weights going **into** neuron $j$, so `W` is `(n_out, n_in)`. Computing every $z_j$ at once for one example is a matrix–vector product:

$$z = W a + b$$

### A batch of examples

We don't feed one example at a time: `x` holds `batch_size` examples, one per **row**, so it's `(batch_size, n_in)`. You want the same thing for every row: row $i$ of the result is $W x_i + b$, the $z$ of example $i$. So the result is `(batch_size, n_out)`: one row per example, one column per neuron.

Careful: $W a$ is written for one example as a column vector. Your examples are **rows**. Same math, but the shapes have to be arranged differently.

### Why forward stores x

Look at page 1 of the notes: $\frac{\partial z}{\partial w} = a^{(L-1)}$. To compute the gradient of the weights, backward will need the input that came in. Backward runs later, so forward has to keep it: store it in `self.x`.

## Your task

In `Linear.forward(self, x)`:

1. store `x` in `self.x`;
2. compute $z$ for every example and return it.

## Shapes

| | Shape |
|---|---|
| `x` (in) | `(batch_size, n_in)` |
| `self.W` | `(n_out, n_in)` |
| `self.b` | `(n_out,)` |
| `z` (out) | `(batch_size, n_out)` |

## Hints

<details>
<summary>Hint 1: start from the shapes</summary>

You need to turn `(batch_size, n_in)` into `(batch_size, n_out)`. With `@`, the inner dims must match and disappear: `(a, k) @ (k, b)` → `(a, b)`. So `x` must be multiplied by something shaped `(n_in, n_out)`. What do you have that can be shaped like that?

</details>

<details>
<summary>Hint 2: the bias</summary>

After the product you have `(batch_size, n_out)` and `b` is `(n_out,)`. A plain `+` adds `b` to every row: that's exactly "every example gets the same bias".

</details>

## Check

```bash
uv run tour.py
```

Next: [step 2, `Linear.backward`](02-linear-backward.md).
