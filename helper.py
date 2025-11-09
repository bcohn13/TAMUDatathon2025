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

# Training loop (pseudo code - fill with your environment interaction code)
def train_dqn(env, num_episodes=1000):
    episodeArray = []
    roundsSurvivedArray = []
    device = torch.device("cpu")
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    replay_buffer = ReplayBuffer()
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    target_update_freq = 10

    for episode in range(num_episodes):
        state = env.reset()  # should return 18x20 grid state (e.g., numpy array)
        done = False
        numberRoundsSurvived = 0

        while not done:
            action = select_action(state, policy_net, epsilon, device)
            next_state, reward, done = env.step(action)  # implement your env's step
            numberRoundsSurvived += 1

            replay_buffer.push(state, next(k for k, v in action_map.items() if v == action), reward, next_state, done)
            state = next_state

            '''
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
            '''
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        #print(f"Episode {episode}, Epsilon {epsilon:.3f}, Survived Rounds {numberRoundsSurvived}")
        episodeArray.append(episode)
        roundsSurvivedArray.append(numberRoundsSurvived)
    plt.plot(episodeArray, roundsSurvivedArray)
    plt.show()
# Note: You need to implement your custom env class with reset() and step(action) methods.
# The state returned should be an 18x20 numpy array representing the grid.
# The step(action) should return next_state, reward, done flag.

