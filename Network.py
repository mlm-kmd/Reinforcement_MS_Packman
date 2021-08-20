import random
import numpy as np
import tensorflow as tf
import os
import keras
import keras.layers
from os import path
#from tensorflow.keras import applications
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from keras.optimizers import RMSprop, Adam

class DDQN_Agent:
    def __init__(self, state_size, action_size, new_model, model_name, crop_size, episode):
        self.state_size = state_size
        self.action_size = action_size
        self.crop_size = crop_size
        self.memory = deque()
        
        # Hyperparameters
        self.gamma = 0.9                                    # Discount rate
        self.epsilon = 5.0 if new_model == True else 0.1    # Exploration rate
        self.epsilon_min = 0.05                             # Minimal exploration rate (epsilon-greedy)
        self.epsilon_decay = 0.00002                        # Decay rate for epsilon
        
        self.model = self._build_model()
        if(new_model == False and path.isdir(model_name)):
            print('loaded in')
            self.model = self.load(model_name, self.model, episode)
        self.model.summary()

    def _build_model(self):
        model = None
        if False:
            #model = applications.Xception(
            #    include_top=True,
            #    weights=None,
            #    input_tensor=None,
            #    input_shape=self.state_size,
            #    pooling="max",
            #    classes=self.action_size,
            #    classifier_activation="relu",
            #)
            model.compile(loss='mse', optimizer=Adam(lr=0.00025))
        else:

            model = Sequential()
            
            # Conv Layers       VGG16 conv layer winner model at 2014
            model.add(Conv2D(filters=32,kernel_size=(3,3), padding='same', activation='relu', input_shape=(self.state_size[0],self.state_size[1]-self.crop_size,self.state_size[2])))
            #model.add(Conv2D(filters=3,kernel_size=(3,3),padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2)))

            model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
            #model.add(Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu"))
            #model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

            #model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
            #model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
            #model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
            #model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

            #model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
            #model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
            #model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
            #model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

            #model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            #model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            #model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            #model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            
            model.add(Flatten())

            #model.add(BatchNormalization())

            # FC Layers
            model.add(Dense(256, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            #model.add(Dense(self.action_size, activation='relu'))
            #model.add(Dense(self.action_size, activation='swish'))
            
            model.compile(loss='mse', optimizer=Adam(lr=0.0005, clipnorm=1.0))
        
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = self.memory#random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)))
            else:
                break
                
            target_f = self.model.predict(state)
            
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0, batch_size=None)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.memory.clear()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name, model, episode):
        model = keras.models.load_model(os.getcwd()+"/"+name+"/TrainedModel_E_"+str(episode)+".h5")
        return model

    def save(self, name, episode):
        if not os.path.exists(os.getcwd()+"/"+name+""):
            os.makedirs(os.getcwd()+"/"+name+"")
        self.model.save(os.getcwd()+"/"+name+"/TrainedModel_E_"+str(episode)+".h5")