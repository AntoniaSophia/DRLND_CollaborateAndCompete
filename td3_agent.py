import numpy as np
import random
from collections import namedtuple, deque

from model_td3 import Actor, Critic
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from FifoMemory import FifoMemory

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 20        # learning timestep interval
LEARN_NUM = 10          # number of learning passes
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter
EPSILON = 1.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 1e-6    # decay rate for noise process

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
class AgentTD3():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, agents_size,  random_seed):
        """Initialize an Agent object.

        Params
        ======
                state_size (int): dimension of each state
                action_size (int): dimension of each action
                random_seed (int): random seed
        """

        # Store parameters
        self.agents_size = agents_size
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = EPSILON


        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(self.agents_size, random_seed)

        # Replay memory
        self.memory = FifoMemory(BUFFER_SIZE, BATCH_SIZE)
        # Short term memory contains only 1/100 of the complete memory and the most recent samples
        self.memory_short = FifoMemory(int(BUFFER_SIZE/100), int(BATCH_SIZE))

    def step(self, state, action, reward, next_state, done, timestep, agent_number):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        self.memory_short.add(state, action, reward, next_state, done)

        # Learn at defined interval, if enough samples are available in memory
        # HINT from Udacity "benchmark": learn every 20 timesteps and train 10 samples
        if len(self.memory) > BATCH_SIZE and timestep % LEARN_EVERY == 0:
            for _ in range(LEARN_NUM):
                experiences = self.memory.sample() 
                experiences_short = self.memory_short.sample() 

                # delay update of the policy and only update every 2nd training
                self.learn(experiences_short, timestep % 2,GAMMA, agent_number)
                self.learn(experiences, timestep % 2 , GAMMA, agent_number)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.agents_size, self.action_size))
        

        self.actor_local.eval()
        with torch.no_grad():
            #action = self.actor_local(state).cpu().data.numpy()
            # TODO: Comment start
            for agent_number, state in enumerate(states):
                #print("State = ", states)

                action = self.actor_local(state).cpu().data.numpy()
                actions[agent_number, :] = action
            # TODO: Comment end

        self.actor_local.train()

        # TD3 --> Action noise regularisation
        if add_noise:
            action += self.epsilon * self.noise.sample()

        # The range of noise is clipped in order to keep the target value 
        # close to the original action.
        clipped_action = np.clip(action, -1, 1)
        
        return clipped_action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, delay, gamma , agent_number):
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
        #print("Actions_Next = " , actions_next.shape)
        #print("actions.shape = " , actions.shape)

        # TODO: Comment start
        # Remark this implementation is not working with more than 2 agents, evil hardcoding...
        if agent_number == 0:
            actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)
        else:
            actions_next = torch.cat((actions[:,:2], actions_next), dim=1)
        # TODO: Comment end
         
        Q_targets_next1, Q_targets_next2 = self.critic_target(next_states, actions_next)

        # TD3 --> Take the minimum of both critic in order to avoid overestimation
        Q_targets_next = torch.min(Q_targets_next1, Q_targets_next2)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected1, Q_expected2 = self.critic_local(states, actions)

        # compute critic loss [HOW MUCH OFF?] as sum of both loss from target
        critic_loss = F.mse_loss(Q_expected1, Q_targets)+F.mse_loss(Q_expected2, Q_targets)

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

            # TODO: comment start
            # Remark this implementation is not working with more than 2 agents, evil hardcoding...
            if agent_number == 0:
                actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
            else:
                actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)
            # TODO: comment end

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
        self.epsilon -= EPSILON_DECAY
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
        self.size = size
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
        #dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.x_previous = x + dx
        return self.x_previous
