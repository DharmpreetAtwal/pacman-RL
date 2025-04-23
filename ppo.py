import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

from torch.distributions import Categorical
import numpy as np
from pprint import pprint

from dqn import state_to_features
from screen import capture_game

# Hyperparameters
gamma = 0.95  # Discount factor
lr_actor = 0.0001  # Actor learning rate
lr_critic = 0.0001  # Critic learning rate
clip_ratio = 0.1  # PPO clip ratio
epochs = 10  # Number of optimization epochs
batch_size = 64  # Batch size for optimization

# Actor and Critic networks
class ActorCritic(nn.Module):
    def __init__(self, action_size, input_shape):
        super(ActorCritic, self).__init__()

        # Add flatten layer to handle image inputs
        self.flatten = nn.Flatten()

        # If input_shape is provided, calculate flattened size
        self.input_shape = input_shape

        # For image inputs: channels, height, width
        flattened_size = input_shape[0] * input_shape[1] * input_shape[2]

        self.shared_input = nn.Linear(flattened_size, 512)

        self.actor_1 = nn.Linear(512, 1024)
        self.actor_2 = nn.Linear(1024, 2048)
        self.actor_3 = nn.Linear(2048, 2048)
        self.policy_logits = nn.Linear(2048, action_size)

        self.critic_1 = nn.Linear(512, 1024)
        self.critic_2 = nn.Linear(1024, 2048)
        self.critic_3 = nn.Linear(2048, 2048)
        self.value = nn.Linear(2048, 1)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, state):
        x = self.flatten(state)
        x = F.relu(self.shared_input(x))

        a = F.relu(self.actor_1(x))
        a = self.dropout(a)
        a = F.relu(self.actor_2(a))
        a = self.dropout(a)
        a = F.relu(self.actor_3(a))
        logits = self.policy_logits(a)

        c = F.relu(self.critic_1(x))
        c = self.dropout(c)
        c = F.relu(self.critic_2(c))
        c = self.dropout(c)
        c = F.relu(self.critic_3(c))
        value = self.value(c)

        return logits, value

# class ActorCritic(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(ActorCritic, self).__init__()
#
#         self.shared_input = nn.Linear(state_size, 512)
#
#         self.actor_1 = nn.Linear(512, 1024)
#         self.actor_2 = nn.Linear(1024, 2048)
#         self.actor_3 = nn.Linear(2048, 2048)
#         self.policy_logits = nn.Linear(2048, action_size)
#
#         self.critic_1 = nn.Linear(512, 1024)
#         self.critic_2 = nn.Linear(1024, 2048)
#         self.critic_3 = nn.Linear(2048, 2048)
#         self.value = nn.Linear(2048, 1)
#
#         self.dropout = nn.Dropout(p=0.2)
#
#
#     def forward(self, state):
#         x = F.relu(self.shared_input(state))
#
#         a = F.relu(self.actor_1(x))
#         a = self.dropout(a)
#         a = F.relu(self.actor_2(a))
#         a = self.dropout(a)
#         a = F.relu(self.actor_3(a))
#         logits = self.policy_logits(a)
#
#         c = F.relu(self.critic_1(x))
#         c = self.dropout(c)
#         c = F.relu(self.critic_2(c))
#         c = self.dropout(c)
#         c = F.relu(self.critic_3(c))
#         value = self.value(c)
#
#         return logits, value

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
        entropy_coef = 0.2

        total_loss = policy_loss + 0.5 * value_loss - (entropy_coef * entropy_bonus)
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


def train_policy_ppo(env, x, y, width, height,  max_episodes=20):
    state = env.reset()

    model = ActorCritic(4, state.shape)
    optimizer = optim.Adam(model.parameters(), lr=lr_actor)

    for episode in range(max_episodes):
        states, actions, rewards, values, returns = [], [], [], [], []
        state = env.reset()

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

            state = next_state

            if done:
                # img = capture_game(x, y, width, height)
                # resized_pil = Image.fromarray(img)
                # resized_pil.show()

                print("---------------------------")
                print(sum(rewards), rewards)
                print(len(rewards))

                if len(rewards) < 50:
                    print("SKIPPED: ", episode)
                    break

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

    torch.save(model.state_dict(), './ppo.pt')


