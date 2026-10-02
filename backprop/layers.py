"""Layers are split into Linear + Sigmoid so each piece can be written and tested on its own.

Following the convention, we call x the input of a given layer, whatever produced it.

Work through this file top to bottom:
    Step 1  Linear.__init__, Linear.forward     lessons/01-linear-forward.md
    Step 2  Linear.backward                     lessons/02-linear-backward.md
    Step 3  Sigmoid.__init__, Sigmoid.forward   lessons/03-sigmoid-forward.md
    Step 4  Sigmoid.backward                    lessons/04-sigmoid-backward.md

Check your progress at any time with:  uv run tour.py
"""

import numpy as np


class Linear:
    """A fully connected layer: every input k feeds every neuron j, z = W·x + b.

    The rest of the code (the checks, and Sequential.step later) looks for these names:
        self.W, self.b     the parameters
        self.dW, self.db   their gradients, filled in by backward
    """

    def __init__(self, n_in, n_out, rng=None):
        """Step 1. Creates the parameters of the layer.

        Args:
            n_in:  how many inputs each example has
            n_out: how many neurons this layer has
            rng:   a numpy random generator, like np.random.default_rng(seed), or None

        Questions to guide you:
            - W[j, k] is w_jk, the weight from input k to neuron j. What shape is W? And b?
            - What would happen if every weight started with the same value?
            - Two layers built from the same seed must be identical. Where should your random numbers come from?
        """
        raise NotImplementedError("Step 1: read lessons/01-linear-forward.md")

    def forward(self, x):
        """Step 1. Computes z = W·x + b for every example in the batch.

        Args:
            x: (batch_size, n_in), one example per row

        Returns:
            z: (batch_size, n_out), one row per example, one column per neuron

        Questions to guide you:
            - W is (n_out, n_in) and x is (batch_size, n_in). How do they line up to give (batch_size, n_out)?
            - Look at the formulas of backward (step 2). Does something seen here need to be cached?
        """
        raise NotImplementedError("Step 1: read lessons/01-linear-forward.md")

    def backward(self, grad_out):
        """Step 2. Answers one question: given how the cost changes with respect to this layer's output z,
        how does it change with respect to everything that produced z?

        Args:
            grad_out: dC/dz, (batch_size, n_out)

        Returns:
            dC/dx, (batch_size, n_in): what gets sent down to the previous layer

        Questions to guide you:
            - Which gradients will the learning step need, and where will it look for them?
            - A gradient has the shape of what it's the gradient of. What shapes are dW and db?
            - W and b are shared by every example of the batch, x is not. What does that change?
            - What does this method need that it doesn't receive as an argument?
        """
        raise NotImplementedError("Step 2: read lessons/02-linear-backward.md")


class Sigmoid:
    """The non-linearity: a = σ(z), applied to every number on its own."""

    def __init__(self):
        """Step 3. Sets up the layer.

        Questions to guide you:
            - Does a sigmoid have anything to learn?
            - Will forward have something to keep for backward? (Read step 4 before you decide.)
        """
        raise NotImplementedError("Step 3: read lessons/03-sigmoid-forward.md")

    def forward(self, z):
        """Step 3. Applies the sigmoid function to z, element by element.

        Args:
            z: any shape, usually (batch_size, n_out) coming from a Linear layer

        Returns:
            σ(z), same shape as z

        Questions to guide you:
            - Does your code work on a whole array at once, or on a single number?
            - Look at σ'(z) in step 4. Is something computed here worth keeping?
        """
        raise NotImplementedError("Step 3: read lessons/03-sigmoid-forward.md")

    def backward(self, grad_out):
        """Step 4. Returns how the cost reacts to the input z of this layer.

        Args:
            grad_out: dC/da, same shape as the output of forward

        Returns:
            dC/dz, same shape as grad_out

        Questions to guide you:
            - Neuron j's output only depends on its own z_j. Is there anything to sum over?
            - σ'(z) can be written with σ(z) itself. Where can you get σ(z) from?
        """
        raise NotImplementedError("Step 4: read lessons/04-sigmoid-backward.md")
