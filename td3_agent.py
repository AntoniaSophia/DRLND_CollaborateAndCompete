import numpy as np
import random
from collections import namedtuple, deque

from td3_model import Actor, Critic
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from FifoMemory import FifoMemory

BUFFER_SIZE = 1000000  # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.95            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-5         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 1       # L2 weight decay
LEARN_EVERY = 1        # learning timestep interval
LEARN_NUM_MEMORY = 1          # number of learning passes
LEARN_NUM_MEMORY_SUCCESS = 3 # number of learning passes of success memory
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter
EPSILON = 5.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 0.995    # decay rate for noise process

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# TD3 = Twin Delayed Deep Deterministic Policy Gradient (TD3)
# 
# Addressing Function Approximation Error in Actor-Critic Methods
# https://arxiv.org/abs/1802.09477
# 
# TD3 description is taken from:
# https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93
# 
# The base for implementation is taken from:
# https://github.com/prasoonkottarathil/Twin-Delayed-DDPG-TD3-/blob/master/TD3.ipynb
#
# TD3 is the successor to the Deep Deterministic Policy Gradient 
# (DDPG)(Lillicrap et al, 2016). 
# Up until recently, DDPG was one of the most used algorithms 
# for continuous control problems such as robotics and autonomous driving. 
# Although DDPG is capable of providing excellent results, it has its drawbacks. 
# Like many RL algorithms training DDPG can be unstable and heavily reliant on 
# finding the correct hyper parameters for the current task (OpenAI Spinning Up, 2018).
# This is caused by the algorithm continuously over estimating the Q values of 
# the critic (value) network. These estimation errors build up over time and 
# can lead to the agent falling into a local optima or experience catastrophic 
# forgetting. TD3 addresses this issue by focusing on reducing the overestimation 
# bias seen in previous algorithms. This is done with the addition of 3 key features:
#    - Using a pair of critic networks (The twin part of the title)
#    - Delayed updates of the actor (The delayed part)
#    - Action noise regularisation (This part didn’t make it to the title :/ )
class TD3Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed=1):
        """Initialize an Agent object.

        Params
        ======
                state_size (int): dimension of each state
                action_size (int): dimension of each action
                random_seed (int): random seed
        """

        # Store parameters
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = EPSILON
        self.lr_actor = LR_ACTOR
        self.lr_critic = LR_CRITIC
        self.lr_decay = WEIGHT_DECAY
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        self.timestep = 0
        # Replay memory
        self.memory = FifoMemory(BUFFER_SIZE, BATCH_SIZE)
        # Short term memory contains only 1/100 of the complete memory and the most recent samples
        self.memory_success = FifoMemory(int(BUFFER_SIZE), int(BATCH_SIZE))
        self.memory_short = FifoMemory(10, 10)

    def update_model(self, state, action, reward, next_state, done):
        self.step(state, action, reward, next_state, done)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.timestep += 1

        reached = True
        if len(self.memory_success) < BATCH_SIZE:
            reached = False

        self.memory.add(state, action, reward, next_state, done)
        self.memory_short.add(state, action, reward, next_state, done)

        if reward > 0.0:
            for i in range(len(self.memory_short)):
                self.memory_success.add(
                    self.memory_short.samples[i].state, \
                    self.memory_short.samples[i].action, \
                    self.memory_short.samples[i].reward, \
                    self.memory_short.samples[i].next_state, \
                    self.memory_short.samples[i].done)
                    
            self.memory_short.clear()

        if reached == False and len(self.memory_success) > BATCH_SIZE:
            print("Success memory ready for use!")


        if len(self.memory) > BATCH_SIZE:
            for i in range(LEARN_NUM_MEMORY):
                experiences = self.memory.sample() 
                # delay update of the policy and only update every 2nd training
                self.learn(experiences, 0 , GAMMA)

        if (len(self.memory_success) > self.memory_success.batch_size):
            for i in range(LEARN_NUM_MEMORY_SUCCESS):
                experiences_success = self.memory_success.sample() 
                self.learn(experiences_success, 0 ,GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # TD3 --> Action noise regularisation
        if add_noise:
            action += self.epsilon * self.noise.sample()

        # The range of noise is clipped in order to keep the target value 
        # close to the original action.
        clipped_action = np.clip(action, -1, 1) 
        
        self.epsilon *= EPSILON_DECAY

        return clipped_action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, delay ,gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value

        Params
        ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
                gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # TD3 --> Using a pair of critic networks (The twin part of the title)

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next1, Q_targets_next2 = self.critic_target(next_states, actions_next)

        # TD3 --> Take the minimum of both critic in order to avoid overestimation
        #Q_targets_next = torch.min(Q_targets_next1, Q_targets_next2)
        Q_targets_next = Q_targets_next1

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected1, Q_expected2 = self.critic_local(states, actions)

        # compute critic loss [HOW MUCH OFF?] as sum of both loss from target
        #critic_loss = F.mse_loss(Q_expected1, Q_targets)+F.mse_loss(Q_expected2, Q_targets)
        critic_loss = F.mse_loss(Q_expected1, Q_targets)
        # minimize loss [TRAIN]
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # TD3 --> Delayed updates of the actor = policy (The delayed part)
        # Compute actor loss
        if delay == 0:
            actions_pred = self.actor_local(states)

            # compute loss [HOW MUCH OFF?]
            actor_loss = -self.critic_local.Q1(states, actions_pred).mean()
            
            # minimize loss [TRAIN]
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

        # ---------------------------- update noise ---------------------------- #
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
                local_model: PyTorch model (weights will be copied from)
                target_model: PyTorch model (weights will be copied to)
                tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=OU_THETA, sigma=OU_SIGMA):
        """Initialize parameters and noise process.
        Params
        ======
                mu: long-running mean
                theta: the speed of mean reversion
                sigma: the volatility parameter
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.x0 = None
        self.x_previous = None
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        if self.x0 is not None:
            self.x_previous = self.x0  
        else:
            self.x_previous = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.x_previous
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.x_previous = x + dx
        return self.x_previous
