# Start here

You are going to write a small neural network, a **multi-layer perceptron**, using nothing but numpy. No PyTorch, no autograd: every gradient is computed by code you write. At the end it learns XOR.

The point is **backpropagation**. Each step gives you the theory first. Then you write the methods of that step: only their signatures are there, the rest is yours. The tour checks your work.

## How it works

1. Read the lesson for the step you're on.
2. Open the file it points to, find the function, replace `raise NotImplementedError(...)` with your code.
3. Run the tour:

   ```bash
   uv run tour.py
   ```

   It checks every step in order, stops at the first one that isn't done, and tells you what's wrong in plain words.

Each lesson ends with hints, hidden behind a click. Try without them first.

Read the lessons rendered, so the math displays and the hints stay folded: on GitHub, or in your editor's Markdown preview (in VS Code: `Ctrl+Shift+V`, or `Cmd+Shift+V` on a Mac).

## The route

You work through three files, top to bottom:

| Step | File | You write | Lesson |
|---|---|---|---|
| 1 | `backprop/layers.py` | `Linear.__init__`, `Linear.forward` | [01](01-linear-forward.md) |
| 2 | `backprop/layers.py` | `Linear.backward` | [02](02-linear-backward.md) |
| 3 | `backprop/layers.py` | `Sigmoid.__init__`, `Sigmoid.forward` | [03](03-sigmoid-forward.md) |
| 4 | `backprop/layers.py` | `Sigmoid.backward` | [04](04-sigmoid-backward.md) |
| 5 | `backprop/losses.py` | `MSE.__init__`, `MSE.forward` | [05](05-mse-forward.md) |
| 6 | `backprop/losses.py` | `MSE.backward` | [06](06-mse-backward.md) |
| 7 | `backprop/network.py` | `Sequential.__init__`, `Sequential.forward` | [07](07-sequential-forward.md) |
| 8 | `backprop/network.py` | `Sequential.backward` | [08](08-sequential-backward.md) |
| 9 | `backprop/network.py` | `Sequential.step` | [09](09-sequential-step.md) |
| 10 | – | nothing: train it on XOR | [10](10-train-xor.md) |

## The big picture

A network is a chain of simple functions. For one layer $L$:

$$z^{(L)} = w^{(L)} a^{(L-1)} + b^{(L)} \qquad a^{(L)} = \sigma\left(z^{(L)}\right)$$

The output of the last layer goes into a **cost** $C$ that measures how wrong the network is (the code calls it the **loss**, in `losses.py`: same thing). Learning means changing every $w$ and $b$ so that $C$ goes down, and for that you need to know **how sensitive $C$ is to each of them**: $\frac{\partial C}{\partial w}$, $\frac{\partial C}{\partial b}$.

Backpropagation is the chain rule, organised so that every piece only does its own small part:

- **forward**: data flows up the chain, each piece computes its output and remembers what it needs;
- **backward**: the gradient flows down the chain. Each piece receives "how $C$ changes with my output" and turns it into "how $C$ changes with my input", which it hands to the piece below.

That's why the code is split into small classes that each have a `forward` and a `backward`.

![The pipeline Linear, Sigmoid, Linear, Sigmoid, MSE: values x, z(1), a(1), z(2), a(2), C flow forward on top in blue, gradients dC/dx to dC/da(2) flow backward below in orange, and the highlighted middle Linear receives how C changes with its output and hands down how C changes with its input.](../assets/lessons/00-big-picture.png)

## Notation: from the math to the code

| Math | Code | Shape |
|---|---|---|
| $a^{(L-1)}$, the input of a layer | `x` | `(batch_size, n_in)` |
| $w_{jk}$, weight from input $k$ to neuron $j$ | `W[j, k]` | `W` is `(n_out, n_in)` |
| $b_j$, bias of neuron $j$ | `b[j]` | `b` is `(n_out,)` |
| $z^{(L)}$ | `z` | `(batch_size, n_out)` |
| $a^{(L)} = \sigma(z^{(L)})$ | what `Sigmoid.forward` returns | `(batch_size, n_out)` |
| $y$, the desired output | `y` | `(batch_size, n_out)` |
| $\frac{\partial C}{\partial (\text{output})}$, the gradient arriving from above | `grad_out` | shaped like the output |

**One row per example.** The network processes a whole batch of examples at once, stacked as the rows of `x`. Row `i` is example `i`. That's why almost every array starts with `batch_size`.

## Three rules that keep you on track

1. **Write down the shapes.** Before writing a line, write down the shape of every array you have and the shape you need to return. The docstrings give you all of them.
2. **A gradient has the shape of what it's the gradient of.** $\frac{\partial C}{\partial W}$ is shaped like `W`, $\frac{\partial C}{\partial b}$ like `b`, $\frac{\partial C}{\partial x}$ like `x`. It's one number per entry: "how much does C move if I nudge this entry?"
3. **When two arrays must be combined, the shapes usually allow only one way.** If you have `(5, 4)` and `(5, 3)` and need `(4, 3)`, there's essentially one product that does it.

## The numpy you need (and nothing more)

| Code | What it does | Shapes |
|---|---|---|
| `A @ B` | matrix product: multiplies and **sums over the shared inner dim** | `(a, k) @ (k, b)` → `(a, b)` |
| `A.T` | transpose: swaps rows and columns | `(a, b)` → `(b, a)` |
| `A * B`, `A + B`, `A - B`, `A / B` | element by element | same shape in, same shape out |
| `A + v` | adds the vector `v` to **every row** of `A` (broadcasting) | `(a, b) + (b,)` → `(a, b)` |
| `np.sum(A, axis=0)` | sums down the rows: one total per column | `(a, b)` → `(b,)` |
| `np.sum(A)` | sums everything into one number | `(a, b)` → `()` |
| `np.exp(A)` | $e^x$ for every entry | same shape |
| `rng.standard_normal(shape)` | random numbers from a generator `rng = np.random.default_rng(seed)` | `shape` |
| `A.shape` | the shape: print it whenever in doubt | |

The checks use sizes that are all different (`batch_size=5`, `n_in=3`, `n_out=4`, plus `hidden=6` for the small networks of steps 7 to 9), so when the tour says something has shape `(4, 5)`, you can read it as `(n_out, batch_size)` and know right away what went wrong.

## Companion notes

[`Backprop.pdf`](../Backprop.pdf) holds the handwritten notes this course grew out of: the chain rule worked out for one neuron, then for many. The lessons refer to them, but each lesson stands on its own.

Ready? Go to [step 1](01-linear-forward.md).
