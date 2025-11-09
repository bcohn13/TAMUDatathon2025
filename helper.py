import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from case_closed_game import Direction
import matplotlib.pyplot as plt

action_map = {
            0: Direction.UP,
            1: Direction.DOWN,
            2: Direction.LEFT,
            3: Direction.RIGHT
}
# Neural network for DQN - input size = 18x20 flattened
class DQN(nn.Module):
    def __init__(self, height=18, width=20, n_actions=4):
        super(DQN, self).__init__()
        self.flatten_size = height * width
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, n_actions)  # outputs Q-value per action
    
    def forward(self, x):
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

# Helper function to select action (epsilon-greedy)
def select_action(state, policy_net, epsilon, device): #head location 
    '''headPos is a tuple(x, y)'''
    if random.random() < epsilon:
        #print("Hello, we are picking a explorative")
        # Random action chosen from available actions
        action_index = random.choice(list(action_map.keys())) #it will never randomly choose to move into a filled wall 
        #while (action_map[action_index] == "UP" and state[headPos[0]][headPos[1]+1] == 1)
            
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action_index = q_values.argmax(1).item()

    # Map the integer action back to environment expected string
    return action_map[action_index]
