import copy
from collections import deque
from tqdm import tqdm
import numpy as np
import torch
import random

from cartpole import Cartpole, Agent

class Transition():
    def __init__(self, s, a, r, sp) -> None:
        self.state = s
        self.action = a
        self.reward = r
        self.next_state = sp

class QLearning(Agent):
    def __init__(self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int=24,
        buffer_length: int=1000,
        eps_decay: int=10000,
        use_gpu: bool=False,
        ) -> None:
        """
        Args:
            state_dim: state dimension for the environment
            action_dim: action dimension for the environment
            hidden_dim: dimension of the hidden layers of the policy network
            buffer_length: max number of samples to store in replay buffer
            eps_decay: increase to slow the rate of epsilon decrease for
                       epsilon-greedy exploration
            use_gpu: whether to use GPU
        """
        super().__init__(state_dim, action_dim, name="q_learning", use_gpu=use_gpu)
        self.hidden_dim = hidden_dim
        self.policy_network = self.build_network().to(self.device)
        self.initial_state_dict = copy.deepcopy(self.policy_network.state_dict())
        self.buffer_length = buffer_length
        self.buffer = deque([], maxlen=buffer_length) # empty replay buffer with the 1000 most recent transitions
        self.iteration = 0
        self.eps_decay = eps_decay

    def reset(self):
        """Reset model, buffer, and iteration count."""
        self.policy_network.load_state_dict(self.initial_state_dict)
        self.buffer = deque([], maxlen=self.buffer_length)
        self.iteration = 0

    def eps_threshold(self) -> float:
        """
        Compute epsilon threshold for eps-greedy exploration that decreases
        as the iteration count increases.
        """
        eps_start, eps_end = 0.9, 0.05
        self.iteration += 1
        return eps_end + (eps_start - eps_end) * np.exp(-1 * self.iteration / self.eps_decay)

    def build_network(self) -> torch.nn.Module:
        """
        Construct a Multi-Layer Perceptron that takes as input the state vector
        and outputs the Q-value vector Q(s, a) for the state. This network has
        3 layers (2 hidden layers) with ReLU activation.

        Returns:
            Network model.
        """
        ##### YOUR CODE STARTS HERE #####
        # Hint: use torch.nn.Sequential
        # Hint: use self.state_dim, self.hidden_dim and self.action_dim
        raise NotImplementedError("Need to implement code here.")
        ###### YOUR CODE ENDS HERE ######

    def policy(self, state: np.ndarray) -> np.ndarray:
        """
        Evaluate policy at the given state.

        Args:
            state: state to evaluate policy

        Returns:
            Optimal action to take from the state.
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(
                self.policy_network(state.to(self.device)), dim=1).numpy()

    def policy_train(self, state: np.ndarray) -> torch.Tensor:
        """
        Evaluate policy at the given state. This will be used during
        the training process in self.train(). Uses an eps-greedy
        approach.

        Args:
            state: state to evaluate policy

        Returns:
            Optimal action to take from the state.
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        ##### YOUR CODE STARTS HERE #####
        # Hint: use random.random() and self.eps_threshold() for eps-greedy
        # Hint: torch.randint, torch.argmax, and torch.no_grad are useful
        raise NotImplementedError("Need to implement code here.")
        ###### YOUR CODE ENDS HERE ######

    def compute_target(self, reward: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """
        Compute target Q-value, reward if terminal (next_state is None) or
        reward + gamma * max_a Q(next_state, a) otherwise.
        """
        ##### YOUR CODE STARTS HERE #####
        # Hint: use torch.no_grad and torch.max
        raise NotImplementedError("Need to implement code here.")
        ###### YOUR CODE ENDS HERE ######

    def add_to_buffer(self, state, action, reward, next_state) -> None:
        """Add a transition to the buffer."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor([reward])
        if next_state is not None:
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        self.buffer.append(Transition(state, action, reward, next_state))

    def sample_buffer(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample replay buffer.

        Returns:
            States, actions, and targets for the minibatch.
        """
        assert len(self.buffer) > self.batch_size
        samples = random.sample(self.buffer, self.batch_size)

        states, actions, targets = [], [], []
        for i in range(self.batch_size):
            s = samples[i].state.to(self.device)
            a = samples[i].action.item()
            r = samples[i].reward.to(self.device)
            sp = None if samples[i].next_state is None else samples[i].next_state.to(self.device)
            states.append(s)
            actions.append(a)
            targets.append(self.compute_target(r, sp))
        
        return torch.cat(states), \
               torch.tensor(actions, dtype=torch.int64).to(self.device).unsqueeze(1), \
               torch.cat(targets).unsqueeze(1)

    def train(self, cartpole: Cartpole, num_episodes: int=100) -> None:
        """
        Train action-value network using deep Q-learning approach described in Algorithm 1 
        from "Playing Atari with Deep Reinforcement Learning".

        Args:
            cartpole: cart-pole environment to train in
            num_episodes: number of episodes to run
        """
        self.reset() # reset model params, replay buffer, epsilon-greedy status
        cartpole.env.reset(seed=0)

        # Define optimizer and loss function
        optimizer = torch.optim.AdamW(self.policy_network.parameters(), 
            lr=self.learning_rate)
        loss_function = torch.nn.MSELoss()

        reward_history = []
        avg_q_history = []
        eval_batch = None
        for episode in tqdm(range(num_episodes)):
            # Initialize episode by resetting the environment
            state, _ = cartpole.env.reset()
            total_reward = 0
            terminated, truncated = False, False

            # Run episode
            while not terminated and not truncated:
                ##### YOUR CODE STARTS HERE #####
                # Perform the following steps here:
                # 1) Evaluate the policy with epsilon greedy with self.policy_train()
                # 2) Use cartpole.env.step to compute next_state, reward, terminated, and truncated
                # 3) Add the transition to the buffer using self.add_to_buffer()
                raise NotImplementedError("Need to implement code here.")
                ###### YOUR CODE ENDS HERE ######

                state = next_state
                total_reward += reward

                # If the buffer is not yet big enough, skip
                if len(self.buffer) <= self.batch_size:
                    eval_batch = copy.deepcopy(self.buffer)
                    continue

                ##### YOUR CODE STARTS HERE #####
                # Sample a mini-batch of transitions from the buffer using self.sample_buffer()
                # then compute the `loss` using loss_function defined above.
                # Hint: when evaluating self.policy_network() on the minibatch of states,
                # you can use gather(1, actions) to get the elements according to the sampled
                # actions.
                raise NotImplementedError("Need to implement code here.")
                ###### YOUR CODE ENDS HERE ######

                # Run a gradient descent step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(
                    self.policy_network.parameters(), 
                    100
                )
                optimizer.step()

            # Store reward for the episode
            reward_history.append(total_reward)

            # For training status visualization, we store the average of the max
            # Q-value for a mini batch of states, which we should see steadily
            # increasing over the episodes
            value = 0
            with torch.no_grad():
                for sample in eval_batch:
                    state = sample.state.to(self.device)
                    value += torch.max(self.policy_network(state)) / len(eval_batch)
            avg_q_history.append(value)

        self.plot_rewards(avg_q_history, ylabel='Avg Max Q')
        self.plot_rewards(reward_history)