# Step 2: `Linear.backward`

> **Write:** `Linear.backward` in `backprop/layers.py`
>
> **Notes:** pages 1, 2 and 3

This is the heart of backpropagation. Take your time.

## The question backward answers

> Given how the cost changes with respect to this layer's output $z$, how does it change with respect to everything that produced $z$?

Something above this layer already worked out how the cost reacts to $z$. That's `grad_out`: entry `[i, j]` is $\frac{\partial C}{\partial z_j}$ for example $i$. It's `(batch_size, n_out)`, one number per entry of $z$.

You don't need to know what's above (a sigmoid, the loss, ten more layers). That's the whole trick: each layer only does its own local part of the chain rule.

From `grad_out`, backward computes three things:

| What | Where it goes | Why |
|---|---|---|
| $\frac{\partial C}{\partial W}$ | `self.dW` | to update the weights (step 9) |
| $\frac{\partial C}{\partial b}$ | `self.db` | to update the biases (step 9) |
| $\frac{\partial C}{\partial x}$ | returned | sent down to the previous layer, so it can do the same |

The third one is the idea of backpropagation from page 2 of the notes: how sensitive the cost is to the **previous layer**.

## One example

In the notes, the chain for one weight of the last layer was:

$$\frac{\partial C_0}{\partial w^{(L)}} = \frac{\partial z^{(L)}}{\partial w^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}} \cdot \frac{\partial C_0}{\partial a^{(L)}}$$

The last two factors, $\sigma'(z) \cdot 2(a - y)$, together say how the cost reacts to $z$. **That's `grad_out`**. For a hidden layer it's a longer product, one factor per piece above it, but it means the same thing, and someone above already computed it. This layer only adds its own factor: how $z$ reacts to $w$, to $b$ and to $a^{(L-1)}$.

Now with many neurons: $z_j = \sum_k w_{jk} a_k + b_j$.

**Weights.** $w_{jk}$ only appears in $z_j$, multiplied by $a_k$:

$$\frac{\partial C}{\partial w_{jk}} = \frac{\partial z_j}{\partial w_{jk}} \cdot \frac{\partial C}{\partial z_j} = a_k \cdot \frac{\partial C}{\partial z_j}$$

**Biases.** $b_j$ only appears in $z_j$, with a factor of 1:

$$\frac{\partial C}{\partial b_j} = 1 \cdot \frac{\partial C}{\partial z_j}$$

**Inputs.** Here's the catch from page 3 of the notes: $a_k$ feeds **every** neuron $j$ of this layer. Each neuron influences all the neurons after it, so you add up all the paths:

$$\frac{\partial C}{\partial a_k} = \sum_j \frac{\partial z_j}{\partial a_k} \cdot \frac{\partial C}{\partial z_j} = \sum_j w_{jk} \cdot \frac{\partial C}{\partial z_j}$$

## A batch of examples

Now add the example index $i$. To keep the formulas short, write:

| Math | Code | Meaning |
|---|---|---|
| $g_{ij}$ | `grad_out[i, j]` | $\frac{\partial C}{\partial z_j}$ for example $i$ |
| $x_{ik}$ | `x[i, k]` | input $k$ of example $i$ |
| $w_{jk}$ | `W[j, k]` | weight from input $k$ to neuron $j$ |

The **weights and biases are shared** by every example, so every example contributes to their gradient. Add them up over $i$:

$$\frac{\partial C}{\partial w_{jk}} = \sum_i g_{ij} x_{ik} \qquad\qquad \frac{\partial C}{\partial b_j} = \sum_i g_{ij}$$

The **inputs are not shared**: row $i$ of `x` belongs to example $i$ only. No sum over examples, just the sum over neurons from before:

$$\frac{\partial C}{\partial x_{ik}} = \sum_j g_{ij} w_{jk}$$

> **Sum, not average?** Page 2 of the notes says the total cost is the **average** over the examples, $\frac{1}{n}\sum$. That $\frac{1}{n}$ is already inside `grad_out`: the loss puts it there (you'll write it in step 6). Divide again here and you divide twice.

## Your task

Write `Linear.backward`. It returns $\frac{\partial C}{\partial x}$, and leaves $\frac{\partial C}{\partial W}$ and $\frac{\partial C}{\partial b}$ in `self.dW` and `self.db`: that's where the learning step (step 9) will look for them.

Questions to ask yourself:

- The formulas need $x$, which isn't an argument of backward. Where does it come from?
- Should backward change `W` and `b`, or only say how they should change?

## Shapes

A gradient has the shape of the thing it's the gradient of:

| | Shape |
|---|---|
| `grad_out` (in) | `(batch_size, n_out)` |
| `x`, the input of forward | `(batch_size, n_in)` |
| `W` | `(n_out, n_in)` |
| `self.dW` | `(n_out, n_in)`, like `W` |
| `self.db` | `(n_out,)`, like `b` |
| returned $\frac{\partial C}{\partial x}$ | `(batch_size, n_in)`, like `x` |

The formulas for $W$ and $x$ are "multiply, then sum over one index". That's exactly what `@` does: `(a, k) @ (k, b)` multiplies and sums over `k`. So for each of them, ask: **which index is summed over**, and which ones are left? The bias is simpler: nothing to multiply, just a sum.

## Hints

<details>
<summary>Hint 1: dW</summary>

$\frac{\partial C}{\partial w_{jk}} = \sum_i g_{ij} x_{ik}$. The sum runs over $i$, the batch dim, and $j$, $k$ are left. For `@` to sum over $i$, it has to be the inner dim: the left factor must be `(n_out, batch_size)` and the right one `(batch_size, n_in)`.

</details>

<details>
<summary>Hint 2: db</summary>

Add up `grad_out` over the examples: one total per neuron. Which axis of `grad_out` is the batch?

</details>

<details>
<summary>Hint 3: the returned gradient</summary>

$\frac{\partial C}{\partial x_{ik}} = \sum_j g_{ij} w_{jk}$. The sum runs over $j$, the neurons. `grad_out` is `(batch_size, n_out)` and `W` is `(n_out, n_in)`: look at the inner dims.

</details>

## Check

```bash
uv run tour.py
```

The checks don't compare against a stored answer: they nudge every entry of `W`, `b` and `x` by a tiny amount, measure how much the cost moves, and compare that with your gradient. If they agree, your calculus is right.

Next: [step 3, `Sigmoid.forward`](03-sigmoid-forward.md).
