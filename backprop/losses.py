"""The cost function: how wrong is the network?

    Step 5  MSE.__init__, MSE.forward   lessons/05-mse-forward.md
    Step 6  MSE.backward                lessons/06-mse-backward.md

Check your progress at any time with:  uv run tour.py
"""

import numpy as np


class MSE:
    """Squared error, summed over the output neurons and averaged over the examples."""

    def __init__(self):
        """Step 5. Sets up the loss.

        Questions to guide you:
            - Does the loss have anything to learn?
            - backward() takes no arguments. Where will it find what it needs?
        """
        raise NotImplementedError("Step 5: read lessons/05-mse-forward.md")

    def forward(self, pred, y):
        """Step 5. The loss returns one number only: the cost, averaged over all the examples.

        Args:
            pred: a^(L), the network output, (batch_size, n_out)
            y:    the desired output,        (batch_size, n_out)

        Returns:
            C, a single number

        Questions to guide you:
            - What do you divide by: the number of examples, or the number of entries?
            - backward() will receive nothing. What will it need from this call?
        """
        raise NotImplementedError("Step 5: read lessons/05-mse-forward.md")

    def backward(self):
        """Step 6. dC/da^(L): where backpropagation starts.

        Returns:
            dC/da^(L), same shape as pred: (batch_size, n_out)

        Questions to guide you:
            - Each entry a_ij appears in exactly one term of C. What's the derivative of that term?
            - Where does the 1/n of the average end up?
        """
        raise NotImplementedError("Step 6: read lessons/06-mse-backward.md")
