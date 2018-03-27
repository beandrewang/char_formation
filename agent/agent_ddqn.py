import random
import numpy as np
from collections import deque
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, add, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras import backend as K
from keras import regularizers
import pickle

BATCH_SIZE = 32
EPISODES = 2500000
EPISODE_LENGTH = 500

class DQNAgent:
    def __init__(self, n_state_width, n_state_height, n_actions, epsilon=1.0, train = True):

        self.memory = deque(maxlen=20000)

        self.n_actions = n_actions
        self.n_state_width = n_state_width
        self.n_state_height = n_state_height
        self.learning_rate = 0.0001

        # Discount factor
        self.gamma = 0.9

        # Exploration rate and decay
        self.epsilon = epsilon
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.03
        self.train = train
        self.model = self.create_resnet()
        self.target_model = self.create_resnet()

    def create_resnet(self):
        inputs = Input(shape=(self.n_state_height, self.n_state_width, 3))

        # conv1
        conv1 = Conv2D(32, (3, 3), padding = 'same', activation='relu')(inputs)

        # conv2
        x = Conv2D(32, (3, 3), padding = 'same', activation='relu')(conv1)

        # conv3
        x = Conv2D(32, (3, 3), padding = 'same', activation='relu')(x)

        # add
        x = add([x, conv1])
        # full connection Dense
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(self.n_actions, activation='linear')(x)

        model = Model(inputs=inputs, outputs=x)
        
        '''
        for layer in model.layers[:-2]:
            layer.trainable = False
        '''

        model.summary()
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate, decay = 1e-6))

        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_model(self, weights):
        self.model.set_weights(weights)

    def read_model(self):
        return self.model.get_weights()

    def get_action(self, state, train = True):
        """
        Perform an action given environment state.
        :param state: Discrete environment state (integer)
        :return: Action to be performed (integer)
        """
        if train:
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, self.n_actions)
            else:
                action_values = self.model.predict(state)
                best_action = np.argmax(action_values[0])
        else:
            action_values = self.model.predict(state)
            best_action = np.argmax(action_values[0])
        #print action_values, best_action
        return best_action

    def evaluate(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, terminated in batch:
            final_target = self.model.predict(state)
            if not terminated:
                # target = immediate reward + (discount factor * value of next state)
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                # if it's a terminal state, the value of this state equals to immediate reward
                target = reward
            final_target[0][action] = target
            return self.model.evaluate(state, final_target)



    def experience_replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        total_loss = 0
        for state, action, reward, next_state, terminated in batch:
            # Predict value for a current state
            final_target = self.model.predict(state)
            if not terminated:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                #target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                target = reward + self.gamma * t[np.argmax(a)]
            else:
                # if it's a terminal state, the value of this state equals to immediate reward
                target = reward
            final_target[0][action] = target
            #total_loss += self.model.evaluate(state, final_target)
            self.model.fit(state, final_target, epochs=1, verbose=0)
        # Decrease exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss / batch_size

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def read_epsilon(self):
        return self.epsilon

    def load(self, name):
        """
        Load saved Value Function Approximation weights.
        :param name: Model filename.
        """
        self.model.load_weights(name)

    def save(self, name):
        """
        Save Value Function Approximation weights.
        :param name: Model filename.
        """
        self.model.save_weights(name)
