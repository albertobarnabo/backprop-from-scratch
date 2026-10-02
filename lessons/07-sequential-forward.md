# Step 7: `Sequential.__init__` and `Sequential.forward`

> **Write:** `Sequential.__init__` and `Sequential.forward` in `backprop/network.py`

## Theory

A network is a chain of layers. The output of one layer is the input of the next:

$$x \to \text{Linear} \to \text{Sigmoid} \to \text{Linear} \to \text{Sigmoid} \to a^{(2)}$$

Written as one formula, that's a nested function:

$$a^{(2)} = \sigma\Big(W^{(2)} \sigma\big(W^{(1)} x + b^{(1)}\big) + b^{(2)}\Big)$$

In code, `Sequential` holds the layers in a list, in the order the data flows through them. Its `forward` feeds the input to the first layer, that output to the second layer, and so on, and returns what comes out of the last one.

It doesn't need to know what kind of layers they are: each one has a `forward`, and that's all it uses.

While the data flows up, every layer keeps what its own backward will need. That's why forward always runs before backward.

## Your task

Write `Sequential.__init__` and `Sequential.forward`, which passes `x` through every layer, in order, and returns the output of the last one.

Questions to ask yourself:

- What will forward, backward and step need to find later?
- What does each layer receive as its input?

## Shapes

| | Shape |
|---|---|
| `x` (in) | `(batch_size, n_in)` of the first layer |
| returned | `(batch_size, n_out)` of the last layer |

The shape changes along the way: each Linear changes the number of columns, each Sigmoid keeps it.

![The XOR network as a chain of arrays drawn as grids: x (4, 2) goes through Linear(2, 4) to z^(1) (4, 4), Sigmoid to a^(1) (4, 4), Linear(4, 1) to z^(2) (4, 1) and Sigmoid to a^(2) (4, 1); a caption says all 4 examples are in one batch, one row per example, so the 4 rows stay the same and only the columns change at each Linear.](../assets/lessons/07-network-shapes.png)

## Hints

<details>
<summary>Hint 1</summary>

A `for` loop over the layers, where each layer's output becomes the value you pass to the next one.

</details>

## Check

```bash
uv run tour.py
```

Next: [step 8, `Sequential.backward`](08-sequential-backward.md).
