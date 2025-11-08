import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

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
def select_action(state, policy_net, epsilon, device):
    if random.random() < epsilon:
        return random.randint(0, 3)  # random action (UP, DOWN, LEFT, RIGHT)
    else:
        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0).to(device)  # add batch dim
            q_values = policy_net(state)
            return q_values.argmax(1).item()

# Training loop (pseudo code - fill with your environment interaction code)
def train_dqn(env, num_episodes=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        while not done:
            action = select_action(state, policy_net, epsilon, device)
            next_state, reward, done = env.step(action)  # implement your env's step

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay_buffer) > batch_size:
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

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Epsilon {epsilon:.3f}")

# Note: You need to implement your custom env class with reset() and step(action) methods.
# The state returned should be an 18x20 numpy array representing the grid.
# The step(action) should return next_state, reward, done flag.

