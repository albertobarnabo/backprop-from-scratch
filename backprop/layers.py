"""Layers are split into Linear + Sigmoid so each piece can be written and tested on its own.

Following the convention, we call x the input of a given layer, whatever produced it.

Work through this file top to bottom:
    Step 1  Linear.forward     lessons/01-linear-forward.md
    Step 2  Linear.backward    lessons/02-linear-backward.md
    Step 3  Sigmoid.forward    lessons/03-sigmoid-forward.md
    Step 4  Sigmoid.backward   lessons/04-sigmoid-backward.md

Check your progress at any time with:  uv run tour.py
"""

import numpy as np


class Linear:
    """A fully connected layer: every input k feeds every output neuron j."""

    def __init__(self, n_in, n_out, rng=None):
        """Already written for you. Takes a Random Number Generator (rng) to give a seed for reproducibility.

        W[j, k] is w_jk, the weight from input k to neuron j  ->  W is (n_out, n_in)
        b[j]    is b_j,  the bias of neuron j                 ->  b is (n_out,)
        """
        rng = rng or np.random.default_rng(0)

        self.W = rng.standard_normal((n_out, n_in)) / np.sqrt(n_in)
        self.b = np.zeros(n_out)

        self.x = None   # forward() stores its input here: backward() needs it
        self.dW = None  # backward() stores dC/dW here
        self.db = None  # backward() stores dC/db here

    def forward(self, x):
        """Step 1. Compute z = W·x + b for every example in the batch.

        Args:
            x: (batch_size, n_in), one example per row

        Returns:
            z: (batch_size, n_out), one row per example, one column per neuron

        Careful about the dims: W is (n_out, n_in) and x is (batch_size, n_in).
        """
        raise NotImplementedError("Step 1: read lessons/01-linear-forward.md")

    def backward(self, grad_out):
        """Step 2. Answers one question: given how the cost changes with respect to this
        layer's output z, how does it change with respect to everything that produced z?

        Args:
            grad_out: dC/dz, (batch_size, n_out)

        Returns:
            dC/dx, (batch_size, n_in): what gets sent down to the previous layer

        Also stores:
            self.dW: dC/dW, (n_out, n_in)
            self.db: dC/db, (n_out,)
        """
        raise NotImplementedError("Step 2: read lessons/02-linear-backward.md")


class Sigmoid:
    """The non-linearity: a = σ(z), applied to every number on its own."""

    def __init__(self):
        self.sig_z = None  # forward() stores σ(z) here: backward() needs it

    def forward(self, z):
        """Step 3. Applies the sigmoid function to z, element by element.

        Args:
            z: any shape, usually (batch_size, n_out) coming from a Linear layer

        Returns:
            σ(z), same shape as z
        """
        raise NotImplementedError("Step 3: read lessons/03-sigmoid-forward.md")

    def backward(self, grad_out):
        """Step 4. Returns how the cost reacts to the input z of this layer.

        Args:
            grad_out: dC/da, same shape as the output of forward

        Returns:
            dC/dz, same shape as grad_out
        """
        raise NotImplementedError("Step 4: read lessons/04-sigmoid-backward.md")
