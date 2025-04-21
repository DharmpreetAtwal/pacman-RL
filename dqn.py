import random

import numpy as np
import torch
from torch import optim, nn

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(Policy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),

            nn.Linear(2048, output_dim)
        )

    def forward(self, x):
        return self.network(x)  # Returns Q-values directly


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(np.array(done))
        )

    def __len__(self):
        return len(self.buffer)

def train_policy_dqn(env, num_episodes=100, lr=0.0001):
    state = env.reset()
    state = state_to_features(state)
    policy = Policy(len(state), env.action_space.n)

    target_network = Policy(len(state_to_features(env.reset())), env.action_space.n)
    target_network.load_state_dict(policy.state_dict())
    replay_buffer = ReplayBuffer()

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    epsilon = 0.2
    gamma = 0.99
    batch_size = 64

    for episode in range(num_episodes):
        state_dict = env.reset()
        state = state_to_features(state_dict)

        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, env.action_space.n - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy(state_tensor)
                    action = q_values.max(1)[1].item()

            # Take action
            next_state_dict, reward, done = env.step(action)
            next_state = state_to_features(next_state_dict)

            total_reward += reward
            # print(reward)

            # Store in replay buffer
            replay_buffer.push(state, action, reward / 1000.0, next_state, int(done))

            # next state
            state = next_state

            # if we have enough samples, train with replay buffer
            if len(replay_buffer) > batch_size:
                # Sample from replay buffer
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(
                    batch_size)

                # Compute Q values
                current_q = policy(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                # Compute target Q values
                with torch.no_grad():
                    next_q = target_network(batch_next_states).max(1)[0]
                    target_q = batch_rewards + gamma * next_q * (1 - batch_dones)

                # Compute loss
                loss = nn.MSELoss()(current_q, target_q)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network every 10 episodes
        if episode % 10 == 0:
            target_network.load_state_dict(policy.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

def state_to_features(state_dict, normalize=True):
    # Define normalization constants
    max_pos_x, max_pos_y = 512, 512
    max_lives = 5
    max_mode = 3

    # Initialize feature vector
    features = []

    # Process pacman position
    pacman_pos = state_dict['pacman_position']
    if normalize:
        features.append(pacman_pos[0] / max_pos_x)
        features.append(pacman_pos[1] / max_pos_y)
    else:
        features.append(pacman_pos[0])
        features.append(pacman_pos[1])

    # Process pacman lives
    lives = state_dict['pacman_lives']
    if normalize:
        features.append(lives / max_lives)
    else:
        features.append(lives)

    # Process ghost positions and modes
    for ghost in ['inky', 'blinky', 'pinky', 'clyde']:
        ghost_pos = state_dict[f'{ghost}_position']
        if normalize:
            features.append((ghost_pos[0] + 512) / (max_pos_x + 512))
            features.append((ghost_pos[1] + 512) / (max_pos_y + 512))
        else:
            features.append(ghost_pos[0])
            features.append(ghost_pos[1])

        ghost_mode = state_dict[f'{ghost}_mode']
        if normalize:
            features.append(ghost_mode / max_mode)
        else:
            features.append(ghost_mode)

    features.extend([state_dict["pellet_above"], state_dict["pellet_bottom"],
                     state_dict["pellet_left"], state_dict["pellet_right"],])

    # print(features)
    return features

