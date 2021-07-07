import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, n_layers=0, n_inputs=None):
        """
        Deep Neural Network model to be used for linear regression tasks by
        Regressor

        Args:
            n_layers {int}: number of layers, excluding output layer
            n_inputs {int}: number of inputs for each layer, includes input of
                output layer
        """
        super(Network, self).__init__()

        # Random seed for this network
        torch.manual_seed(42)

        layers = []

        # Create input and hidden layers
        for index in range(n_layers):
            layers.append(nn.Linear(n_inputs[index],
                                    n_inputs[index + 1]).double())

        # Create output layer
        layers.append(nn.Linear(n_inputs[-1], 1).double())

        # Add layers as individual parameters of Network as per sklearn
        # estimator convension
        self.layers = nn.ModuleList(layers)

    def forward(self, x, activations, dropout=False):
        """
        Overwrites nn.Module function forward(), perform a single forward pass
        of the model

        Args:
            x {torch.tensor}: input values
            activations {List[nn.modules.activation]}: List of activation
                functions, note that the output layer does not have an
                activation function, it is linear.
            dropout {boolean}: whether dropout is enabled on this forward pass

        Returns: {torch.tensor} output values

        """
        result = x
        for i, layer in enumerate(self.layers[:-1]):
            if not dropout:
                result = activations[i](layer(result))
            else:
                result = nn.Dropout()(activations[i](layer(result)))
        result = self.layers[-1](result)
        return result
