"""The cost function: how wrong is the network?

    Step 5  MSE.forward    lessons/05-mse-forward.md
    Step 6  MSE.backward   lessons/06-mse-backward.md

Check your progress at any time with:  uv run tour.py
"""

import numpy as np


class MSE:
    """Squared error, summed over the output neurons and averaged over the examples."""

    def __init__(self):
        self.pred = None  # forward() stores the prediction here: backward() needs it
        self.y = None     # forward() stores the desired output here: backward() needs it

    def forward(self, pred, y):
        """Step 5. The loss returns one number only: the cost, averaged over all the examples.

        Args:
            pred: a^(L), the network output, (batch_size, n_out)
            y:    the desired output,        (batch_size, n_out)

        Returns:
            C, a single number
        """
        raise NotImplementedError("Step 5: read lessons/05-mse-forward.md")

    def backward(self):
        """Step 6. dC/da^(L): where backpropagation starts.

        Takes no grad_out: the cost sits at the top of the chain, nothing comes after it.

        Returns:
            dC/da^(L), same shape as pred: (batch_size, n_out)
        """
        raise NotImplementedError("Step 6: read lessons/06-mse-backward.md")
