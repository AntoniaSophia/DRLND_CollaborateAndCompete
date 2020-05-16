import random
import torch
import numpy as np
from collections import namedtuple, deque 

# Taken from https://github.com/hengyuan-hu/rainbow/blob/master/core.py

# CPU OR GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Sample(object):
    def __init__(self, state, action, reward, next_state, done):
        self._state = state
        self._next_state = next_state
        self.action = action
        self.reward = reward
        self.done = done

    @property
    def state(self):
        return self._state

    @property
    def next_state(self):
        return self._next_state

    def __repr__(self):
        info = ('S(mean): %3.4f, A: %s, R: %s, NS(mean): %3.4f, Done: %s'
                % (self.state.mean(), self.action, self.reward,
                   self.next_state.mean(), self.done))
        return info


class FifoMemory(object):
    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.samples = []

        self.oldest_idx = 0

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        return len(self.samples)

    def _evict(self):
        """Simplest FIFO eviction scheme."""
        to_evict = self.oldest_idx
        self.oldest_idx = (self.oldest_idx + 1) % self.max_size
        return to_evict

    def add(self, state, action, reward, next_state, end):
        assert len(self.samples) <= self.max_size

        new_sample = Sample(state, action, reward, next_state, end)
        if len(self.samples) == self.max_size:
            avail_slot = self._evict()
            self.samples[avail_slot] = new_sample
        else:
            self.samples.append(new_sample)

    def sample(self):
        """Simpliest uniform sampling (w/o replacement) to produce a batch.
        """
        assert self.batch_size < len(self.samples), 'no enough samples to sample from'

        experiences = random.sample(self.samples, self.batch_size)

        # convert experience tuples to arrays
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def clear(self):
        self.samples = []
        self.oldest_idx = 0
