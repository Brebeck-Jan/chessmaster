from keras.layers import Input, Dense, Flatten, Concatenate, Conv2D, Dropout
from keras.losses import mean_squared_error
from keras.models import Model, clone_model, load_model
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np

class Agent(object):

    def __init__(self):

        # set up model optimizer
        self.optimizer = RMSprop(learning_rate = 0.01) # Choose optimal learning rate!
        
        # set up model
        self.model = Model()

        # build model
        self.init_model()
    
    def fix_model(self):
        """ Creates Model used for bootstrapping. """

        # clone model
        self.fixed_model = clone_model(self.model)

        # compile fixed model
        self.fixed_model.compile(optimizer = self.optimizer, loss = "mse", metrics = ["mae"])

        # set weights
        self.fixed_model.set_weights(self.model.get_weights())

    def init_model(self):
        """ Build neural network. """

        # input is state (of chessboard, in layer form)
        layer_state = Input(shape=(8,8,8), name = "state")

        # set up parameters for Conv2D layers | Find explanation!
        filters = 4
        kernel_size = 1
        activation = "relu"

        # Conc2D Layers | Find explanation for exact implementation!
        conv_01 = Conv2D(filters, (kernel_size, kernel_size), activation = activation)(layer_state)
        conv_02 = Conv2D(2*filters, (2*kernel_size, 2*kernel_size), strides = (1, 1), activation = activation)(layer_state)
        conv_03 = Conv2D(3*filters, (3*kernel_size, 3*kernel_size), strides = (2, 2), activation = activation)(layer_state)
        conv_04 = Conv2D(4*filters, (4*kernel_size, 4*kernel_size), strides = (2, 2), activation = activation)(layer_state)
        conv_05 = Conv2D(5*filters, (5*kernel_size, 5*kernel_size), activation = activation)(layer_state)

        conv_rank = Conv2D(3, (1, 8), activation = activation)(layer_state)
        conv_file = Conv2D(3, (8, 1), activation = activation)(layer_state)

        # Flatten Conv2D Layers
        f_01 = Flatten()(conv_01)
        f_02 = Flatten()(conv_02)
        f_03 = Flatten()(conv_03)
        f_04 = Flatten()(conv_04)
        f_05 = Flatten()(conv_05)

        f_rank = Flatten()(conv_rank)
        f_file = Flatten()(conv_file)

        # Concatenate layers add add more dense layers
        d_activation = "sigmoid"

        dense_01 = Concatenate(name= "dense_base")([f_01, f_02, f_03, f_04, f_05, f_rank,f_file])
        
        dense_02 = Dense(256, activation = d_activation)(dense_01)
        dense_03 = Dense(128, activation = d_activation)(dense_02)
        dense_04 = Dense(56, activation = d_activation)(dense_03)
        dense_05 = Dense(64, activation = d_activation)(dense_04)
        dense_06 = Dense(32, activation = d_activation)(dense_05)

        # final layer
        value_head = Dense(1)(dense_06)

        # set model
        self.model = Model(inputs = layer_state, outputs = value_head)

        # compile model
        self.model.compile(optimizer = self.optimizer, loss = mean_squared_error)

    def predict(self, layer_board):
        """ Predict state vlaue of layer_board. """

        return self.model.predict(layer_board)
    
    def TD_update(self, states, rewards, sucstates, episode_active, gamma = 0.9):
        """
        Update SARSA Network with minibatch samples
        """

        suc_state_values = self.fixed_model.predict(sucstates) # Implement fix model in learn.py!!!!! update after x itertaions!
        V_target = np.array(rewards) + np.array(episode_active) * gamma * np.squeeze(suc_state_values)

        # Train with gradient descent step of minibatch
        self.model.fit(x = states, y = V_target, epochs = 1, verbose = 0)

        # get expected returns
        V_state = self.model.predict(states)

        # get td error
        td_errors = V_target - np.squeeze(V_state)

        return td_errors