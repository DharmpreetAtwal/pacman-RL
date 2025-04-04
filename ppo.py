import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pyexpat import features
from torch.distributions import Categorical
import numpy as np
import gym

from dqn import state_to_features

# Environment setup
# env = gym.make('CartPole-v1')
# state_size = env.observation_space.shape[0]
# action_size = env.action_space.n

# Hyperparameters
gamma = 0.99  # Discount factor
lr_actor = 0.001  # Actor learning rate
lr_critic = 0.001  # Critic learning rate
clip_ratio = 0.2  # PPO clip ratio
epochs = 10  # Number of optimization epochs
batch_size = 64  # Batch size for optimization


# Actor and Critic networks
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.dense1 = nn.Linear(state_size, 64)
        self.policy_logits = nn.Linear(64, action_size)
        self.dense2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.dense1(state))
        logits = self.policy_logits(x)
        value = self.value(x)
        return logits, value


# PPO algorithm
def ppo_loss(model, optimizer, old_logits, old_values, advantages, states, actions, returns):
    def compute_loss(logits, values, actions, returns):
        actions_onehot = F.one_hot(actions, 4).float()
        policy = F.softmax(logits, dim=1)
        action_probs = torch.sum(actions_onehot * policy, dim=1)
        old_policy = F.softmax(old_logits, dim=1)
        old_action_probs = torch.sum(actions_onehot * old_policy, dim=1)

        # Policy loss
        ratio = torch.exp(torch.log(action_probs + 1e-10) - torch.log(old_action_probs + 1e-10))
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))

        # Value loss
        value_loss = torch.mean(torch.square(values - returns))

        # Entropy bonus (optional)
        entropy_bonus = torch.mean(policy * torch.log(policy + 1e-10))

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
        return total_loss

    def get_advantages(returns, values):
        advantages = returns - values
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def train_step(states, actions, returns, old_logits, old_values):
        model.train()
        logits, values = model(states)
        loss = compute_loss(logits, values, actions, returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    advantages = get_advantages(returns, old_values)

    for _ in range(epochs):
        loss = train_step(states, actions, returns, old_logits, old_values)

    return loss


def train_policy_ppo(env, max_episodes=30):
    # Initialize actor-critic model and optimizer
    state_dict = env.reset()
    state = state_to_features(state_dict)
    model = ActorCritic(len(state), 4)
    optimizer = optim.Adam(model.parameters(), lr=lr_actor)

    # if os.path.exists("./test.pt"):
    #     model.load_state_dict(torch.load("./CPS824.pt", weights_only=True))

    for episode in range(max_episodes):
        states, actions, rewards, values, returns = [], [], [], [], []
        state_dict = env.reset()
        state = state_to_features(state_dict)

        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(state_tensor)

            # Sample action from the policy distribution
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value.item())

            state_dict = next_state
            state = state_to_features(state_dict)

            if done:
                # Calculate returns
                returns_batch = []
                discounted_sum = 0
                for r in reversed(rewards):
                    discounted_sum = r + gamma * discounted_sum
                    returns_batch.insert(0, discounted_sum)

                # Convert lists to tensor format
                states_tensor = torch.FloatTensor(np.array(states))
                actions_tensor = torch.LongTensor(np.array(actions))
                values_tensor = torch.FloatTensor(np.array(values))
                returns_tensor = torch.FloatTensor(np.array(returns_batch))

                # Get old logits
                with torch.no_grad():
                    old_logits, _ = model(states_tensor)

                # Calculate advantages
                advantages = returns_tensor - values_tensor

                # Update using PPO
                loss = ppo_loss(model, optimizer, old_logits, values_tensor, advantages,
                                states_tensor, actions_tensor, returns_tensor)

                print(f"Episode: {episode + 1}, Loss: {loss.item()}")

                break

    torch.save(model.state_dict(), './test.pt')