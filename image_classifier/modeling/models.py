"""Module for containing architectures/definition of models."""

import torch


class Net(torch.nn.Module):
    """Simple CNN model for MNIST classification."""

    def __init__(self):
        """
        Initialises the model.

        This function will create all the necessary components for the model (layers,
        activation functions, etc.) and assign them to the object's attributes.
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass of the model.

        This function defines the forward pass of the model. The forward pass is the
        process of computing the output of the model given the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        output : torch.Tensor
            Output tensor of shape (N, C).
        """
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output
