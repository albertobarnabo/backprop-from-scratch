# Step 3: `Sigmoid.__init__` and `Sigmoid.forward`

> **Write:** `Sigmoid.__init__` and `Sigmoid.forward` in `backprop/layers.py`
>
> **Notes:** page 1, where $a^{(L)} = \sigma(z^{(L)})$

## Theory

### Why a non-linearity

Stack two Linear layers without anything in between:

$$W^{(2)} \left(W^{(1)} x + b^{(1)}\right) + b^{(2)} = \left(W^{(2)} W^{(1)}\right) x + \left(W^{(2)} b^{(1)} + b^{(2)}\right)$$

That's just one Linear layer with weights $W^{(2)} W^{(1)}$. However many you stack, the network can only draw straight lines, and XOR can't be separated by a straight line. The non-linear function $\sigma$ between the layers is what lets the network bend.

### The sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

It squashes any number into $(0, 1)$: very negative $z$ gives almost 0, very positive almost 1, and $\sigma(0) = 0.5$.

If numpy ever warns about an *overflow in exp*, don't worry: for a very negative $z$, $e^{-z}$ is too big to store and becomes infinity, and $1 / \infty = 0$ is still the right answer.

It's applied to **every number on its own**: $a_j = \sigma(z_j)$. Neuron $j$'s activation only depends on neuron $j$'s $z$. So the shape never changes.

## Your task

Write `Sigmoid.__init__` and `Sigmoid.forward`, which returns $\sigma(z)$ for every entry of `z`.

Questions to ask yourself:

- A sigmoid has no weights. Does its `__init__` need anything at all?
- Read the theory of step 4 before deciding: is there a value computed in forward that backward will want?

## Shapes

| | Shape |
|---|---|
| `z` (in) | anything, usually `(batch_size, n_out)` |
| returned | same as `z` |

## Hints

<details>
<summary>Hint 1</summary>

No loops needed. `np.exp` works on every entry of an array, and so do `+`, `-` and `/` with a number: `1 / array` divides 1 by every entry.

</details>

## Check

```bash
uv run tour.py
```

Next: [step 4, `Sigmoid.backward`](04-sigmoid-backward.md).
