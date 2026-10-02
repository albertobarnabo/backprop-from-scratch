"""Stacking layers into a network, and learning.

    Step 7  Sequential.__init__, Sequential.forward   lessons/07-sequential-forward.md
    Step 8  Sequential.backward                       lessons/08-sequential-backward.md
    Step 9  Sequential.step                           lessons/09-sequential-step.md

Check your progress at any time with:  uv run tour.py
"""


class Sequential:
    def __init__(self, layers: list):
        """Step 7. A network made of the given layers, in the order the data flows through them.

        Questions to guide you:
            - What will forward, backward and step need to find later?
        """
        raise NotImplementedError("Step 7: read lessons/07-sequential-forward.md")

    def forward(self, x):
        """Step 7. Carries the output of each layer into the next one.

        Args:
            x: the network input, (batch_size, n_in)

        Returns:
            the output of the last layer, (batch_size, n_out)

        Questions to guide you:
            - What does each layer receive as its input?
        """
        raise NotImplementedError("Step 7: read lessons/07-sequential-forward.md")

    def backward(self, grad):
        """Step 8. Backpropagation through the whole network.

        Args:
            grad: dC/da^(L), the gradient coming from the loss, (batch_size, n_out)

        Returns:
            dC/dx for the network input, (batch_size, n_in)

        Questions to guide you:
            - Which layer should receive grad first?
            - What does each layer's backward receive, and what does it give back?
            - The next forward still needs the layers in their original order. Does your loop leave them as they are?
        """
        raise NotImplementedError("Step 8: read lessons/08-sequential-backward.md")

    def step(self, lr):
        """Step 9. One step of gradient descent, with learning rate lr.

        Returns:
            nothing: it changes the layers in place

        Questions to guide you:
            - Which layers have something to learn, and how can you tell them apart?
            - The gradient points uphill. Which way should the parameters move, and by how much?
        """
        raise NotImplementedError("Step 9: read lessons/09-sequential-step.md")
