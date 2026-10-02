# Step 8: `Sequential.backward`

> **Write:** `Sequential.backward` in `backprop/network.py`
>
> **Notes:** page 2 (the idea of backpropagation)

## Theory

### The chain rule, across the whole network

For a network Linear → Sigmoid → Linear → Sigmoid → cost, the chain rule gives the gradient as a product of one factor per piece:

$$\frac{\partial C}{\partial x} = \frac{\partial z^{(1)}}{\partial x} \cdot \frac{\partial a^{(1)}}{\partial z^{(1)}} \cdot \frac{\partial z^{(2)}}{\partial a^{(1)}} \cdot \frac{\partial a^{(2)}}{\partial z^{(2)}} \cdot \frac{\partial C}{\partial a^{(2)}}$$

Read it **right to left**: start from the cost and walk down. The last factor comes from the loss (`MSE.backward`, step 6). Every other factor belongs to one layer, and that layer's `backward` is exactly the code that applies it:

- it receives $\frac{\partial C}{\partial (\text{its output})}$, everything to its right, already multiplied together;
- it applies its own factor and returns $\frac{\partial C}{\partial (\text{its input})}$;
- its input is the previous layer's output, so what it returns is exactly what the previous layer's `backward` needs.

```
forward    x ──> Linear ──> Sigmoid ──> Linear ──> Sigmoid ──> pred ──> MSE ──> C

backward  dx <── Linear <── Sigmoid <── Linear <── Sigmoid <───────────── MSE.backward()
                   │                      │
                 dW, db                 dW, db
```

That's page 2 of the notes: "how sensitive is the cost to the previous layer?" asked once per layer, top to bottom. Along the way every `Linear` stores its `dW` and `db`.

### Where the first gradient comes from

`Sequential.backward(grad)` receives `grad` = $\frac{\partial C}{\partial a^{(L)}}$, the last factor of the formula above (here $L = 2$), from `MSE.backward()`. The network doesn't know about the loss: in the training loop you call them one after the other.

## Your task

Write `Sequential.backward`, which returns $\frac{\partial C}{\partial x}$ for the network input.

Questions to ask yourself:

- Which layer should receive `grad` first?
- What does each layer's backward receive, and what does it give back?
- The next forward still needs the layers in their original order. Does your loop leave them as they are?

## Shapes

| | Shape |
|---|---|
| `grad` (in) | `(batch_size, n_out)` of the last layer |
| returned | `(batch_size, n_in)` of the first layer |

## Hints

<details>
<summary>Hint 1</summary>

`reversed(some_list)` gives you its items from the last to the first, and leaves the list alone. Avoid `some_list.reverse()`: it flips the list itself, so the network would be upside down on the next forward.

</details>

<details>
<summary>Hint 2</summary>

It's the same shape of loop as `forward`, run backwards: each `backward` call's result is the value you pass to the next call.

</details>

## Check

```bash
uv run tour.py
```

This step checks the whole chain: a Linear → Sigmoid → Linear → Sigmoid network with the MSE, every gradient compared with the numerical one.

Next: [step 9, `Sequential.step`](09-sequential-step.md).
