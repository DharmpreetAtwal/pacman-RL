from random import random
from typing import Optional, Tuple

import math
import os
import numpy as np
import pygame
from pygame import Vector2
from pygame.locals import *
from sympy.abc import epsilon

from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
from mazedata import MazeData

import random
import torch
import torch.nn as nn
import torch.optim as optim

import gym
from gym import spaces

class GameController(object):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(False)
        self.level = 0
        #self.lives = 5
        #Try with 1 life
        self.lives = 5
        self.score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.mazedata = MazeData()

        self.done = False
        self.power_left = 4
        self.ghosts_eaten = 0

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level%5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def startGame(self):
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites(self.mazedata.obj.name+".txt", self.mazedata.obj.name+"_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup(self.mazedata.obj.name+".txt")
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart))
        self.pellets = PelletGroup(self.mazedata.obj.name+".txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)

        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)

    def startGame_old(self):
        self.mazedata.loadMaze(self.level)#######
        self.mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup("maze1.txt")
        self.nodes.setPortalPair((0,17), (27,17))
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12,14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15,14), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.pellets = PelletGroup("maze1.txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 0+14))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0+11.5, 3+14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4+11.5, 3+14))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, RIGHT, self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)



    def update(self):
        dt = self.clock.tick(300) / 200.0
        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.ghosts.update(dt)
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt)
        else:
            self.pacman.update(dt)

        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm

        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        self.render()

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        # self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            #self.hideEntities()

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
                self.power_left -= 1
            if self.pellets.isEmpty():
                self.flashBG = True
                self.hideEntities()
                # self.pause.setPause(pauseTime=3, func=self.nextLevel)

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.ghosts_eaten += 1

                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    self.showEntities()
                    # self.pause.setPause(pauseTime=1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -=  1
                        self.lifesprites.removeImage()
                        self.pacman.die()
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVERTXT)
                            self.restartGame()
                            self.done = True
                            # self.pause.setPause(pauseTime=3, func=self.restartGame)
                        else:
                            self.done = False
                            self.resetLevel()
                            # self.pause.setPause(pauseTime=3, func=self.resetLevel)

    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20), self.level)
                print(self.fruit)
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(str(self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time=1)
                fruitCaptured = False
                for fruit in self.fruitCaptured:
                    if fruit.get_offset() == self.fruit.image.get_offset():
                        fruitCaptured = True
                        break
                if not fruitCaptured:
                    self.fruitCaptured.append(self.fruit.image)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        self.pause.paused = False
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def restartGame(self):
        self.lives = 5
        self.level = 0
        self.pause.paused = False
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []

    def resetLevel(self):
        self.pause.paused = False
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def render(self):
        self.screen.blit(self.background, (0, 0))
        #self.nodes.render(self.screen)
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)

        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))

        for i in range(len(self.fruitCaptured)):
            x = SCREENWIDTH - self.fruitCaptured[i].get_width() * (i+1)
            y = SCREENHEIGHT - self.fruitCaptured[i].get_height()
            self.screen.blit(self.fruitCaptured[i], (x, y))

        pygame.display.update()



class PacManEnv(gym.Env):
    def __init__(self, game, initial_pellets):
        self.game = game
        self.initial_pellets = initial_pellets
        self.action_space = spaces.Discrete(4)
        self.last_eaten = -1
        self.observation_space = spaces.Dict({
            "pacman_position": spaces.Box(np.array([16, 64]), np.array([512, 512]), dtype=np.int16),
            "pacman_lives": spaces.Discrete(6),

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
        pellets_left = [(pellet.position.x, pellet.position.y) for pellet in self.game.pellets.pelletList]

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

            "pellets": [1 if pellet in pellets_left else 0 for pellet in self.initial_pellets],
        }

    def _calculate_rewards(self):
        return (2000 * self.game.score) - (10000 * pow(2, 8 - self.game.lives))

    def reset(self):
        self.game.restartGame()
        self.game.resetLevel()

        self.game.done = False
        self.game.power_left = 4
        self.game.ghosts_eaten = 0

        self.last_eaten = -1

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

        pre_action_reward = self._calculate_rewards()


        # Take action, update
        pygame.event.post(action_event)
        self.game.update()

        delta_reward = self._calculate_rewards() - pre_action_reward
        done = len(self.game.pellets.pelletList) == 0 or self.game.done

        if done:
            self.game.done = False
            delta_reward = 0

        if delta_reward == 0:
            delta_reward = self.last_eaten
            self.last_eaten -= 20
        elif delta_reward == -500000:
            self.last_eaten -= 20
        else:
            self.last_eaten = -1

        return self._get_obs(), delta_reward, done

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

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, output_dim)
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

def state_to_features(state_dict, normalize=True):
    # Define normalization constants
    max_pos_x, max_pos_y = 512, 512  # Adjust based on game dimensions
    max_lives = 5
    max_mode = 1

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
            features.append(ghost_pos[0] / max_pos_x)
            features.append(ghost_pos[1] / max_pos_y)
        else:
            features.append(ghost_pos[0])
            features.append(ghost_pos[1])

        ghost_mode = state_dict[f'{ghost}_mode']
        if normalize:
            features.append(ghost_mode / max_mode)
        else:
            features.append(ghost_mode)

    features.extend(state_dict['pellets'])

    return features


def train_policy(env, policy, num_episodes=500, lr=0.001):
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

        # Update target network every few episodes
        if episode % 10 == 0:
            target_network.load_state_dict(policy.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")



if __name__ == "__main__":
    game = GameController()
    game.startGame()

    pellet_list = [(pellet.position.x, pellet.position.y) for pellet in game.pellets.pelletList]
    env = PacManEnv(game, pellet_list)

    state_dict = env.reset()
    input_dim = len(state_to_features(state_dict))
    output_dim = env.action_space.n

    policy = Policy(input_dim, output_dim)

    if os.path.exists("./CPS824.pt"):
        policy.load_state_dict(torch.load("./CPS824.pt", weights_only=True))
    train_policy(env, policy)

    torch.save(policy.state_dict(), './CPS824.pt')
