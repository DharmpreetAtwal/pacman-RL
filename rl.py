import math
import os
import random

import gym
import numpy as np
import pygame
import torch
from gym.vector.utils import spaces
from pygame import KEYDOWN, K_UP, K_RIGHT, K_DOWN, K_LEFT

from dqn import train_policy_dqn, state_to_features, Policy
from pacman import Pacman
from ppo import train_policy_ppo
from run import GameController


class PacManEnv(gym.Env):
    def __init__(self, game, initial_pellets):
        self.post_lives = None
        self.pre_lives = None
        self.post_score = None
        self.pre_score = None

        self.live_reward = 10000

        self.game = game
        self.initial_pellets = initial_pellets
        self.action_space = spaces.Discrete(4)
        self.last_eaten = -1
        self.observation_space = spaces.Dict({
            "pacman_position": spaces.Box(np.array([16, 64]), np.array([512, 512]), dtype=np.int16),
            # "pacman_lives": spaces.Discrete(6),

            "inky_position": spaces.Box(np.array([16, 64]), np.array([512, 512]), dtype=np.int16),
            "inky_mode": spaces.Discrete(2),

            "blinky_position": spaces.Box(np.array([16, 64]), np.array([512, 512]), dtype=np.int16),
            "blinky_mode": spaces.Discrete(2),

            "pinky_position": spaces.Box(np.array([16, 64]), np.array([512, 512]), dtype=np.int16),
            "pinky_mode": spaces.Discrete(2),

            "clyde_position": spaces.Box(np.array([16, 64]), np.array([512, 512]), dtype=np.int16),
            "clyde_mode": spaces.Discrete(2),

            "pellets": spaces.Box(np.array([0 for _ in game.pellets.pelletList]),
                                  np.array([1 for _ in game.pellets.pelletList]), dtype=np.int8),
        })

    def _get_obs(self):
        pellet_surrounding = []
        pacman_pos = (self.game.pacman.position.x, self.game.pacman.position.y)

        for pellet in self.game.pellets.pelletList:
            dist = math.sqrt(math.pow(pellet.position.x - pacman_pos[0], 2) +
                             math.pow(pellet.position.y - pacman_pos[1], 2))
            if dist <= 32:
                pellet_surrounding.append((pellet.position.x, pellet.position.y))

        pellet_above = 0
        pellet_right = 0
        pellet_left = 0
        pellet_bottom = 0

        for pellet in pellet_surrounding:
            if pellet[1] < pacman_pos[1]:
                pellet_above = 1
            elif pellet[1] > pacman_pos[1]:
                pellet_bottom = 1
            elif pellet[0] < pacman_pos[0]:
                pellet_left = 1
            elif pellet[0] > pacman_pos[0]:
                pellet_right = 1

        return {
            "pacman_position": (int(self.game.pacman.position.x), int(self.game.pacman.position.y)),
            "pacman_lives": self.game.lives,

            "inky_position": (int(self.game.ghosts.inky.position.x), int(self.game.ghosts.inky.position.y)),
            "inky_mode": self.game.ghosts.inky.mode.current,

            "blinky_position": (int(self.game.ghosts.blinky.position.x), int(self.game.ghosts.blinky.position.y)),
            "blinky_mode": self.game.ghosts.blinky.mode.current,

            "pinky_position": (int(self.game.ghosts.pinky.position.x), int(self.game.ghosts.blinky.position.y)),
            "pinky_mode": self.game.ghosts.pinky.mode.current,

            "clyde_position": (int(self.game.ghosts.clyde.position.x), int(self.game.ghosts.clyde.position.y)),
            "clyde_mode": self.game.ghosts.clyde.mode.current,

            "pellet_above": pellet_above,
            "pellet_right": pellet_right,
            "pellet_left": pellet_left,
            "pellet_bottom": pellet_bottom,
        }

    def _calculate_rewards(self):
        if self.pre_pellets - self.post_pellets == 0:
            self.live_reward -= 100
        else:
            self.live_reward += 1000

        if self.live_reward < -50000:
            self.live_reward = -10000

        reward = (10000000 * (self.pre_pellets - self.post_pellets)) + self.live_reward

        if self.pre_lives - self.post_lives != 0:
            reward = -2000 * (self.post_pellets * 100)
            self.live_reward = 5000

        return reward

    def reset(self):
        self.game.restartGame()
        self.game.resetLevel()

        self.game.done = False
        self.game.power_left = 4
        self.game.ghosts_eaten = 0

        self.live_reward = 10000

        return self._get_obs()

    def step(self, action):
        action_event = None
        if action == 0:
            action_event = pygame.event.Event(KEYDOWN, {'key': K_UP, 'mod': 0})
        elif action == 1:
            action_event = pygame.event.Event(KEYDOWN, {'key': K_RIGHT, 'mod': 0})
        elif action == 2:
            action_event = pygame.event.Event(KEYDOWN, {'key': K_DOWN, 'mod': 0})
        elif action == 3:
            action_event = pygame.event.Event(KEYDOWN, {'key': K_LEFT, 'mod': 0})
        else:
            raise NotImplementedError()


        self.pre_pellets = len(self.game.pellets.pelletList)
        self.pre_score = self.game.score
        self.pre_lives = self.game.lives

        # Take action, update
        pygame.event.post(action_event)
        self.game.update()

        self.post_pellets = len(self.game.pellets.pelletList)
        self.post_score = self.game.score
        self.post_lives = self.game.lives


        reward = self._calculate_rewards()
        done = len(self.game.pellets.pelletList) == 0 or self.game.done

        return self._get_obs(), reward, done


if __name__ == "__main__":
    game = GameController()
    game.startGame()

    pellet_list = [(pellet.position.x, pellet.position.y) for pellet in game.pellets.pelletList]
    env = PacManEnv(game, pellet_list)

    train_policy_ppo(env)