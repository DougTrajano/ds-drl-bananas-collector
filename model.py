import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, nb_hidden, seed=1412):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            nb_hidden (int): Number of hidden layers
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(state_size, nb_hidden),
            nn.ReLU(),
            nn.Linear(nb_hidden, nb_hidden),
            nn.ReLU(),
            nn.Linear(nb_hidden, action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model(state)