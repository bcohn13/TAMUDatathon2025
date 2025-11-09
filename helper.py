import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from case_closed_game import Game, Direction, GameResult

action_map = {0: Direction.UP, 1: Direction.DOWN, 2: Direction.LEFT, 3: Direction.RIGHT}
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
def select_action(state, policy_net, epsilon, device,action_map):
    if random.random() < epsilon:
        # Random action chosen from available actions
        action_index = random.choice(list(action_map.keys()))
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action_index = q_values.argmax(1).item()

    # Map the integer action back to environment expected string
    return action_map[action_index]

# Training loop (pseudo code - fill with your environment interaction code)
def train_dqn(env, num_episodes=1000):
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

        while not done:
            action = select_action(state, policy_net, epsilon, device,action_map)
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
        return policy_net, target_net
    
class CaseClosedEnv:
    def __init__(self):
        self.game = Game()
    
    def reset(self):
        """Reset the game and return initial state as 18x20 numpy array"""
        self.game.reset()
        return np.array(self.game.board.grid, dtype=np.float32)
    
    def step(self, action: Direction):
        """
        Take an action for agent1. 
        For training, agent2 can be a simple random agent.
        
        Returns:
            next_state: 18x20 numpy array
            reward: float
            done: bool
        """
        # Map numeric action to Direction
        action_map = {
            0: Direction.UP,
            1: Direction.DOWN,
            2: Direction.LEFT,
            3: Direction.RIGHT
        }

         
        agent1_dir = action
        
        # Use random move for agent2
        possible_moves = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        agent2_dir = np.random.choice(possible_moves)
        
        # Step the game
        result = self.game.step(agent1_dir, agent2_dir)
        
        # Reward shaping
        if result == GameResult.AGENT1_WIN:
            reward = 1.0
            done = True
        elif result == GameResult.AGENT2_WIN:
            reward = -1.0
            done = True
        elif result == GameResult.DRAW:
            reward = 0.0
            done = True
        else:
            reward = 0.1  # small reward for surviving
            done = False
        
        next_state = np.array(self.game.board.grid, dtype=np.float32)
        return next_state, reward, done