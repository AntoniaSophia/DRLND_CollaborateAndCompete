import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from ddpg_model import ActorNetwork, CriticNetwork   
from collections import deque, namedtuple
import random
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = np.finfo(np.float32).eps.item()

   
class DDPGAgent:
    """
    A DDPG Agent
    """    
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_actor = 1e-4,
                 lr_critic = 1e-4,
                 lr_decay = .995,
                 replay_buff_size = 1000000,
                 gamma = .95,
                 batch_size = 64,
                 random_seed = 42,
                 soft_update_tau = 1e-3
                 ):
        """
        Initialize model
        """
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.lr_critic = lr_critic
        self.lr_decay = lr_decay
        self.tau = soft_update_tau
        
        self.actor_local = ActorNetwork(state_dim, action_dim).to(device=device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(device=device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)
        
        self.critic_local = CriticNetwork(state_dim, action_dim).to(device=device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(device=device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic)
        
        self.noise = OUNoise(action_dim, random_seed)
        
        self.memory = ReplayBuffer(action_dim, replay_buff_size, batch_size, random_seed)
        
        
    def update_model(self, state, action, reward, next_state, done):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        
        :experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        :gamma (float): discount factor
        """
        self.memory.add(state, action, reward, next_state, done)
        if not self.memory.is_ready():
            return
        
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones)).detach()
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.smooth_l1_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)   
        
    def act(self, state, noise_t=0.0):
        """
        Returns actions for given state as per current policy.
        """
        if len(np.shape(state)) == 1:
            state = state.reshape(1,-1)
        state = torch.from_numpy(state).float().to(device=device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += self.noise.sample() * noise_t
        return np.clip(action, -1, 1).squeeze()
    
    def reset(self):
        self.noise.reset()
        
    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :local_model: PyTorch model (weights will be copied from)
        :target_model: PyTorch model (weights will be copied to)
        :tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            

class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """
        Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """
        Reset the internal state (= noise) to mean (mu).
        """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        Update internal state and return it as a noise sample.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize a ReplayBuffer object.
        
        :buffer_size (int): maximum size of buffer
        :batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    def is_ready(self):
        return len(self.memory) > self.batch_size

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)
        
    
