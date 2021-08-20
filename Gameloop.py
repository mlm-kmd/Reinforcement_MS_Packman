import Environment
import random
import gym
import numpy as np
import Network
import Preprocess
import subprocess
import time as timer
import sys
import pandas as pd
from collections import deque

#remember to set new model to 'false' if your not training a new model, or else epsilon will start reseted and exporation will occour.
env = Environment.environment()
preprocess = Preprocess.preprocesser()


for e in range(env.episodes):
    total_reward = 0
    game_score = 0
    image_timer = 0
    env.total_times_trained += 1
    img = env.env.reset()
    env.image_blend.clear()
    total_lives = 3
    state = preprocess.process_frame(env.env.reset(),env.state_size,env.crop_size)
    
    for skip in range(env.skip_start):
        env.env.step(0)
    
    for time in range(200000):
        image_timer += 1        
        
        action = env.agent.act(state)
        next_state, reward, done, lives = env.env.step(action)
        if total_lives != lives["ale.lives"]:
            total_lives = lives["ale.lives"]
            reward = reward - 100

        game_score += reward
        total_reward += reward

        if time < 5*14:
            continue

        next_state = preprocess.process_frame(next_state, env.state_size, env.crop_size)
        env.agent.remember(state, action, reward, next_state, done)

        state = next_state
        
        env.env.render()

        if done:
            env.all_rewards += game_score
            if game_score > env.highest_reward:
                env.highest_reward = game_score

            if e == env.episodes_save_timer * env.save_done:
                env.agent.save(env.model_name, env.episodes)
                df = pd.DataFrame(data={'total_times_trained': [round(env.total_times_trained)], 'all_rewards': [round(env.all_rewards)], 'highest_reward': [round(env.highest_reward)], 'avg_reward': [round(env.all_rewards / env.total_times_trained)]})
                env.training_logs = env.training_logs.append(df,ignore_index=True)
                env.training_logs.to_csv(env.model_name+"/training_logs.csv",index=False)
                env.save_done += 1

            print("episode: {}/{}, total_times_trained: {}, game score: {}, reward: {}, avg reward: {}"
                  .format(e+1, env.episodes, env.total_times_trained, game_score, total_reward, env.all_rewards/env.total_times_trained))
            break
        env.agent.replay(env.batch_size)