import copy

import torch
import torch.nn as nn

import pickle
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split, \
    cross_val_score
from sklearn.base import BaseEstimator

from part2_network import Network


class Regressor(BaseEstimator):

    def __init__(self,
                 x=None,
                 nb_epoch=500,
                 n_hidden=1,
                 n_nodes=None,
                 activations=None,
                 optimiser=torch.optim.Adam,
                 early_stopping=True,
                 earliest_stop=100,
                 patience=3,
                 dropout=False):
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.
            - n_hidden {int} -- number of layers with activation functions,
                includes the input layer as it requires activation function
            - n_nodes {List[int]} -- number of nodes in each layer except
                input layer, that is generated automatically
            - activations {List[nn.modules.activation]} -- list of activation
                functions for our model
            - optimiser {torch.optim} -- optimiser to be used for the model
            - early_stopping {boolean} -- whether early stopping is enabled for
                traininig the model
            - earliest_stop {int} -- earliest epoch to begin to consier early
                stopping, used to prevent stopping too early before model starts
                converging
            - patience {int} -- number of consecutive epochs without a better
                model before stopping
            - dropout {boolean} -- whether dropout is enabled during training

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self.binariser_labels = None
        self.y_scaler = None
        self.learning_rate = 0.01

        self.x = x
        self.nb_epoch = nb_epoch
        self.n_hidden = n_hidden
        self.n_nodes = n_nodes

        self.output_size = 1
        if x is not None:
            X, _ = self._preprocessor(x, training=True)
            self.input_samples, self.input_size = X.shape
        else:
            self.input_samples = 0
            self.input_size = 13

        # Pre-set or placeholder model variables
        if activations is not None:
            self.activations = activations
        else:
            self.activations = [nn.ReLU() for i in range(self.n_hidden)]

        self.n_inputs = None
        self.model = None
        self.criterion = nn.MSELoss()
        self.optimiser = optimiser

        # Optimisation features
        self.early_stopping = early_stopping
        self.earliest_stop = earliest_stop
        self.patience = patience
        self.dropout = dropout

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _build_optimiser(self):
        """
        Builds an optimiser from the set optimiser types

        Returns: {nn.optim} instantiated optimiser
        """
        # Use given model optimiser
        if self.optimiser == torch.optim.Adam:
            return self.optimiser(self.model.parameters(),
                                  lr=self.learning_rate,
                                  weight_decay=1e-4)

        # Use classic SGD
        return self.optimiser(self.model.parameters(), lr=self.learning_rate)

    def _preprocessor(self, x, y=None, training=False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size 
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size 
                (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Fill NaN values within the numerical columns of the given input data
        numerical_x = x.drop("ocean_proximity", axis=1)
        fill_keys = numerical_x.median().to_dict()
        numerical_x = numerical_x.fillna(value=fill_keys)

        # Normalise numerical values to scale to values between 0 and 1
        numerical_labels = numerical_x.keys()
        x_scaler = MinMaxScaler()
        numerical_x = pd.DataFrame(data=x_scaler.fit_transform(numerical_x),
                                   columns=numerical_labels)

        # Perform one-hot encoding on textual values "ocean_proximity" replace
        # with columns of x_ij in [0, 1] for each label
        binariser = LabelBinarizer()
        ocean_proximity = x["ocean_proximity"]

        if training:
            # Create new binariser parameters for one-hot encoding
            binarised_data = binariser.fit_transform(ocean_proximity)
            self.binariser_labels = binariser.classes_
        else:
            # Use existing binariser parameters for one-hot encoding
            binariser.fit(self.binariser_labels)
            binarised_data = binariser.transform(ocean_proximity)

        # One-hot encoding parameters
        op_frame = pd.DataFrame(data=binarised_data,
                                columns=binariser.classes_)

        # Combine numerical and one-hot encoded textual data frames
        x = pd.concat([numerical_x, op_frame], axis=1)

        # Process on CPU, as Lab computers throw CUDA error
        device = torch.device('cpu')

        # Create tensor from preprocessed x data
        x_tensor = torch.tensor(x.values, device=device, requires_grad=True)
        y_tensor = None

        if isinstance(y, pd.DataFrame):
            # Normalise y values if available, store scaler to be used to undo
            # scaling
            self.y_scaler = MinMaxScaler()
            y = pd.DataFrame(data=self.y_scaler.fit_transform(y),
                             columns=y.keys())
            y_tensor = torch.tensor(y.values, device=device, requires_grad=True)

        # Return preprocessed x and y, return None for y if it was None
        return x_tensor, y_tensor

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # In order to match sklearn estimator API convention, the model is built
        # here instead of in the constructor
        if self.n_nodes is not None:
            self.n_inputs = [self.input_size] + self.n_nodes
        else:
            self.n_inputs = [self.input_size] +\
                        [self.input_size for i in range(self.n_hidden)]

        # Neural network model
        self.model = Network(n_layers=self.n_hidden,
                             n_inputs=self.n_inputs)

        self.optimiser = self._build_optimiser()

        # Split data into train-validation if early stopping
        if self.early_stopping:
            x, val_x, y, val_y = train_test_split(x,
                                                  y,
                                                  train_size=0.8,
                                                  random_state=42)

        # Preprocess training data
        X_train, Y_train = self._preprocessor(x, y=y, training=True)

        if self.early_stopping:
            # Also preprocess validation data
            X_val, Y_val = self._preprocessor(val_x, val_y)

            # Values for determining and storing optimal model
            best_model = None
            min_loss = 999
            strikes = 0

        # Set model mode to train
        self.model.train()
        for epoch in range(self.nb_epoch):
            # Clear gradient for this iteration
            self.optimiser.zero_grad()

            # Forward pass and loss calculation
            y_predicted = self.model(X_train,
                                     self.activations,
                                     dropout=self.dropout)

            loss = self.criterion(y_predicted, Y_train)

            # Backpropagation
            loss.backward()
            self.optimiser.step()

            if self.early_stopping:
                # Predict off validatation data and calculate loss
                val_y_pred = self.model(X_val, self.activations)
                val_loss = self.criterion(val_y_pred, Y_val)
                # print(f'val loss: {val_loss:.5f} best loss: {min_loss:.5f}')

                # Determine if training should stop early
                if val_loss.item() > min_loss:
                    if epoch >= self.earliest_stop:
                        if strikes == self.patience:
                            # Early stop
                            self.model = best_model
                            return self
                        else:
                            strikes += 1
                else:
                    # Clear strikes and save optimal model
                    strikes = 1
                    min_loss = val_loss.item()
                    best_model = copy.deepcopy(self.model)

            if (epoch + 1) % 10 == 0:
                print(f'epoch {epoch + 1} / {self.nb_epoch} loss = {loss.item():.8f}')

        if self.early_stopping:
            # Return optimal model if available
            self.model = best_model
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        if self.model is not None:
            X, _ = self._preprocessor(x, training=False)  # Do not forget

            # Set model to evaluate mode
            self.model.eval()

            # Predict from input and scale result using y scaler from training
            y_predicted = self.model(X, self.activations).detach().numpy()
            y_predicted_scaled = self.y_scaler.inverse_transform(y_predicted)

            return y_predicted_scaled

        return None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        y_predicted = self.predict(x)
        mse = mean_squared_error(y, y_predicted)
        return np.sqrt(mse)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x, y):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        - x {pd.DataFrame} -- Raw input array of shape
            (batch_size, input_size).
        - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1)

    Returns:
        - {dict{str, any}} optimised hyper-parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    param_grid = [
        {
            'n_hidden': [1],
            'n_nodes': [[13], [24], [32]],
            'activations': [
                [nn.Sigmoid()],
                [nn.ReLU()],
                [nn.Tanh()]
            ]
        },
        {
            'n_hidden': [2],
            'n_nodes': [[13, 13], [20, 20], [24, 32], [24, 64]],
            'activations': [
                [nn.Sigmoid(), nn.Sigmoid()],
                [nn.ReLU(), nn.ReLU()],
                [nn.Tanh(), nn.Tanh()]
            ]
        },
        {
            'n_hidden': [3],
            'n_nodes': [[13, 13, 13], [32, 64, 32], [64, 128, 64], [512, 512, 128]],
            'activations': [
                [nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid()],
                [nn.Sigmoid(), nn.ReLU(), nn.ReLU()],
                [nn.ReLU(), nn.ReLU(), nn.ReLU()],
                [nn.ReLU(), nn.Sigmoid(), nn.ReLU()]
            ],
            'patience': [1, 3, 5, 8],
            'dropout': [True, False]
        },
        {
            'n_hidden': [4],
            'n_nodes': [[32, 64, 64, 32]],
            'activations': [
                [nn.Sigmoid(), nn.Tanh(), nn.ReLU(), nn.Sigmoid()],
                [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()]
            ]
        },
    ]

    regressor = Regressor()
    grid_search = GridSearchCV(regressor,
                               param_grid,
                               cv=5,
                               scoring="neg_mean_squared_error")

    grid_search.fit(x, y)

    print(grid_search.best_score_)
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

    cvres = grid_search.cv_results_

    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    # Return the chosen hyper parameters
    return grid_search.best_params_

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Set manual seed for result replication
    torch.manual_seed(42)

    # Shuffle data
    data = shuffle(data, random_state=42)
    data.reset_index(inplace=True, drop=True)

    # Spliting input and output
    x_raw = data.loc[:, data.columns != output_label]
    y_raw = data.loc[:, [output_label]]

    x_train, x_test, y_train, y_test \
        = train_test_split(x_raw, y_raw, test_size=0.2, random_state=42)

    # Training
    # Build Regressor
    acitivations = [nn.ReLU(), nn.ReLU(), nn.ReLU()]
    n_nodes = [512, 512, 128]
    regressor = Regressor(x_train,
                          nb_epoch=500,
                          n_hidden=3,
                          n_nodes=n_nodes,
                          activations=acitivations,
                          patience=5,
                          dropout=False)
    # regressor.fit(x_train, y_train)
    # save_regressor(regressor)

    # Perform cross validation and obtain average error
    nmse_score = -cross_val_score(regressor,
                                  x_raw,
                                  y_raw,
                                  scoring="neg_mean_squared_error",
                                  cv=5)

    scores = np.sqrt(nmse_score)

    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

    # RegressorHyperParameterSearch(x_raw, y_raw)

    # Error
    # error = regressor.score(x_train, y_train)
    # print("\nRegressor error: {}\n".format(error))
    #
    # error = regressor.score(x_test, y_test)
    # print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
