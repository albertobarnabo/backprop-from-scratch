"""Stacking layers into a network, and learning.

    Step 7  Sequential.forward    lessons/07-sequential-forward.md
    Step 8  Sequential.backward   lessons/08-sequential-backward.md
    Step 9  Sequential.step       lessons/09-sequential-step.md

Check your progress at any time with:  uv run tour.py
"""


class Sequential:
    def __init__(self, layers: list):
        """Already written for you. Initializes a network given a list of layers, in the order the data flows through them."""
        self.layers = layers

    def forward(self, x):
        """Step 7. Carries the output of each layer into the next one.

        Args:
            x: the network input, (batch_size, n_in)

        Returns:
            the output of the last layer, (batch_size, n_out)
        """
        raise NotImplementedError("Step 7: read lessons/07-sequential-forward.md")

    def backward(self, grad):
        """Step 8. Backpropagation through the whole network.

        Args:
            grad: dC/da^(L), the gradient coming from the loss, (batch_size, n_out)

        Returns:
            dC/dx for the network input, (batch_size, n_in)

        After this call, every Linear layer has its dW and db filled in.
        """
        raise NotImplementedError("Step 8: read lessons/08-sequential-backward.md")

    def step(self, lr):
        """Step 9. One step of gradient descent: updates W and b of every layer that has
        parameters, using the stored dW and db, scaled by the learning rate lr.

        Returns:
            nothing: it changes the layers in place
        """
        raise NotImplementedError("Step 9: read lessons/09-sequential-step.md")
