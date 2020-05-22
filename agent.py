import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=1412, nb_hidden=128, learning_rate=5e-4, memory_size=int(1e5),
                 prioritized_memory=False, batch_size=64, gamma=0.99, tau=1e-3, small_eps=1e-5, update_every=4):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            nb_hidden (int): Pending
            learning_rate (float): Pending
            memory_size (int): Pending
            prioritized_memory (bool): Pending
            batch_size (int): Pending
            gamma (float): Pending
            tau (float): Pending
            update_every (int): Pending
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.prioritized_memory = prioritized_memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.small_eps = small_eps
        self.update_every = update_every
        self.losses = []

        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, layers=nb_hidden, seed=seed).to(device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, layers=nb_hidden, seed=seed).to(device)       
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

        # Define memory
        if self.prioritized_memory:
            self.memory = PrioritizedMemory(self.memory_size, self.batch_size)
        else:
            self.memory = ReplayMemory(self.memory_size, self.batch_size)
            
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, i):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                if self.prioritized_memory:
                    experiences = self.memory.sample(self.get_beta(i))
                else:
                    experiences = self.memory.sample()
                    
                self.learn(experiences)             

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(int)
        else:
            return random.choice(np.arange(self.action_size)).astype(int)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            small_e (float): 
        """
        if self.prioritized_memory:
            states, actions, rewards, next_states, dones, index, sampling_weights = experiences
            
        else:
            states, actions, rewards, next_states, dones = experiences
            

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        if self.prioritized_memory:
            loss = self.mse_loss_prioritized(Q_expected, Q_targets, index, sampling_weights)
        else:
            loss = F.mse_loss(Q_expected, Q_targets)

        self.losses.append(loss)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_model(self, path):
        torch.save(self.qnetwork_local.state_dict(), path)

    def load_model(self, path):
        self.qnetwork_local.load_state_dict(torch.load(path))
    
    def get_beta(self, i, beta_start=0.4, beta_end=1, beta_growth=1.05):
        if not self.prioritized_memory:
            raise TypeError("This agent is not use prioritized memory")
            
        beta = min(beta_start * (beta_growth ** i), beta_end)
        return(beta)
    
    def mse_loss_prioritized(self, Q_expected, Q_targets, index, sampling_weights):
        losses = F.mse_loss(Q_expected, Q_targets, reduce=False).squeeze(1) * sampling_weights
        self.memory.update_priority(index, losses+self.small_eps)
        return losses.mean()
    
class PrioritizedMemory:
    """
    Fixed-size memory to store experience tuples with sampling weights.
    PRIORITIZED EXPERIENCE REPLAY - https://arxiv.org/pdf/1511.05952.pdf
    """
    
    def __init__(self, memory_size, batch_size, alpha=0.7):
        """Initialize a ReplayMemory object.

        Params
        ======
            memory_size (int): maximum size of memory
            batch_size (int): size of each training batch
            alpha (float): determines how much prioritization is used
        """
        self.capacity = memory_size
        self.memory = deque(maxlen=memory_size)
        self.alpha = alpha
        self.batch_size = batch_size
        self.priority = deque(maxlen=memory_size)
        self.probabilities = np.zeros(memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        priority_max = max(self.priority) if self.memory else 1
        e = self.experience(state, action, reward, next_state, done)
        
        self.memory.append(e)
        self.priority.append(priority_max)
    
    def sample(self, beta=0.4):
        """sample a batch of experiences from prioritized memory."""
        self.update_probabilities()
        index = np.random.choice(range(self.capacity), self.batch_size, replace=False, p=self.probabilities)
        experiences = [self.memory[i] for i in index]
                                 
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        # calculate sampling weights
        sampling_weights = (self.__len__()*self.probabilities[index])**(-beta)
        sampling_weights = sampling_weights / np.max(sampling_weights)
        sampling_weights = torch.from_numpy(sampling_weights).float().to(device)
        
        return (states, actions, rewards, next_states, dones, index, sampling_weights)
    
    def update_probabilities(self):
        probabilities = np.array([i**self.alpha for i in self.priority])
        self.probabilities[range(len(self.priority))] = probabilities
        self.probabilities /= np.sum(self.probabilities)
        
    def update_priority(self, indexes, losses):
        for index, loss in zip(indexes, losses):
            self.priority[index] = loss.data
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
class ReplayMemory:
    """Fixed-size memory to store experience tuples."""

    def __init__(self, memory_size, batch_size):
        """Initialize a ReplayMemory object.

        Params
        ======
            memory_size (int): maximum size of memory
            batch_size (int): size of each training batch
        """
        self.capacity = memory_size
        self.memory = deque(maxlen=memory_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)