import subprocess
import logging
from time import sleep
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from case_closed_game import Direction
import matplotlib.pyplot as plt

from helper import select_action

class DQN(nn.Module):
    def __init__(self, height=18, width=20, n_actions=4):
        super(DQN, self).__init__()
        self.flatten_size = height * width
        self.fc1 = nn.Linear(self.flatten_size, 128) #Creates layers that transforms the data 
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, n_actions)  # outputs Q-value per action (Move: left, right, up, down)
    
    def forward(self, x): #puts the data through the layers 
        x = x.view(-1, self.flatten_size).float()  # flatten input grid
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)
    
# Experience replay buffer using deque
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    
def runEpisode():
    '''Runs the game and returns reward data '''
#Run the game
    python_exe = sys.executable
    subprocess.Popen([python_exe, "agent.py"])
    subprocess.Popen([python_exe,"sample_agent.py"])
    sleep(3)
    subprocess.run([python_exe,"judge_engine.py"])

    #Open the json created by judge_engine
    with open("data.json", "r") as f:
        reward = json.load(f)

    return reward["reward"]

def train_dqn(num_episodes=1000):
    for i in range(num_episodes):
        runEpisode()
    #plt.plot(episodeArray, roundsSurvivedArray)
    #plt.show()

episodeArray = []
roundsSurvivedArray = []
device = torch.device("cpu")
policy_net = DQN().to(device)
target_net = DQN().to(device)
optimizer = optim.Adam(policy_net.parameters())
replay_buffer = ReplayBuffer()
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
target_update_freq = 10

def updateModel(move, state, next_state, episode_num):
    #array with inital starting pos
    global epsilon
    action_map = {0: Direction.UP, 1: Direction.DOWN, 2: Direction.LEFT, 3: Direction.RIGHT}
    #We will instead read a json file created by game to get the states, reward, etc
    with open("data.json", "r") as f:
        data = json.load(f)
    numberRoundsSurvived += 1
    replay_buffer.push(state, next(k for k, v in action_map.items() if v == move), data["reward"], next_state, data["dobe"]) #action is supposed to be directions i think fuuuck why didnt we jut use index

    if len(replay_buffer) > batch_size: #this part is bugged, actions is not ints/nums
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.tensor(states).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)

        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = target_net(next_states).max(1)[0]
        expected_q_values = rewards + gamma * next_q_values * (~dones)

        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()         
           
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode_num % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        #print(f"Episode {episode}, Epsilon {epsilon:.3f}, Survived Rounds {numberRoundsSurvived}")
        
train_dqn(1)
torch.save(policy_net.state_dict(), "policy_net.pth")