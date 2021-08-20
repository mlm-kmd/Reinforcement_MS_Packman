import sys
import random
import gym
import numpy as np
import pandas as pd
import Network
from os import path
from collections import deque

class environment:
    def __init__(self):
        self.env = gym.make('MsPacman-v0')
        self.state_size = (80, 105, 1)
        self.crop_size = 19
        self.action_size = self.env.action_space.n
        self.new_model = True
        self.model_start_episode = 0
        self.model_name = "VGG_Inspired_Model_1.0"

        if path.isfile(self.model_name+"/training_logs.csv"):
            self.training_logs = pd.read_csv(self.model_name+"/training_logs.csv")
        else:
            self.training_logs = pd.DataFrame(data={'total_times_trained': [0],'all_rewards': [0],'highest_reward': [0], 'avg_reward': [0]})

        self.agent = Network.DDQN_Agent(self.state_size, self.action_size, self.new_model, self.model_name, self.crop_size, self.model_start_episode)

        self.episodes = 400
        self.episodes_save_timer = 49
        self.save_done = 1
        self.image_blend = []
        self.batch_size = 1
        self.skip_start = 1
        self.all_rewards = self.training_logs["all_rewards"].iloc[-1]
        self.total_times_trained = self.training_logs["total_times_trained"].iloc[-1]
        self.highest_reward = self.training_logs["highest_reward"].iloc[-1]
        self.done = False