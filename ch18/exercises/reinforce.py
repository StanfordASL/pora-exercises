import copy
from tqdm import tqdm
import numpy as np
import torch

from cartpole import Cartpole, Agent

class Reinforce(Agent):
    def __init__(self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int=24
        ) -> None:
        """
        Args:
            state_dim: state dimension for the environment
            action_dim: action dimension for the environment
            hidden_dim: dimension of the hidden layers of the policy network
        """
        super().__init__(state_dim, action_dim, name="reinforce")
        self.hidden_dim = hidden_dim
        self.policy_network = self.build_network()
        self.initial_state_dict = copy.deepcopy(self.policy_network.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_network.parameters())

    def reset(self):
        """Reset model."""
        self.policy_network.load_state_dict(self.initial_state_dict)

    def build_network(self) -> torch.nn.Module:
        """
        Construct a Multi-Layer Perceptron that takes as input the state vector
        and outputs a vector of action probabilities for the state. This network has
        2 layers with ReLU activation and softmax output.

        Returns:
            Network model.
        """
        ##### YOUR CODE STARTS HERE #####
        # Hint: use torch.nn.Sequential
        # Hint: use self.state_dim, self.hidden_dim and self.action_dim
        # Hint: use torch.nn.Softmax for the output
        raise NotImplementedError("Need to implement code here.")
        ###### YOUR CODE ENDS HERE ######

    def policy(self, state: np.ndarray) -> np.ndarray:
        """
        Evaluate learned stochastic policy.

        Args:
            state: state to evaluate policy

        Returns:
            Action to take from the state.
        """
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            action_probs = self.policy_network(state)
            action_distribution = torch.distributions.Categorical(action_probs)
            return action_distribution.sample().numpy().reshape(1)

    def policy_train(self, state: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate policy at the given state. This will be used during
        the training process in self.train().

        Args:
            state: state to evaluate policy

        Returns:
            Action sampled from the policy distribution, log probability
            of the sampled action.
        """
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        ##### YOUR CODE STARTS HERE #####
        # Hint: see methods of torch.distributions.Categorical
        raise NotImplementedError("Need to implement code here.")
        ###### YOUR CODE ENDS HERE ######

    def sample_episode(self, cartpole : Cartpole) -> tuple[list[float], list[float]]:
        """
        Sample an episode from the environment.

        Args:
            cartpole: cartpole environment

        Returns:
            List of rewards at each timestep.
            List of log probabilities of the actions taken at each timestep.
        """
        log_probs = []
        rewards = []
        terminated, truncated = False, False
        state, _ = cartpole.env.reset()
        while not terminated and not truncated:
            action, log_prob = self.policy_train(state)
            state, reward, terminated, truncated, _ = cartpole.env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
        return rewards, log_probs

    def train(self, 
        cartpole: Cartpole, 
        num_episodes: int=1000, 
        use_causality: bool=False, 
        use_baseline: bool=True,
        ) -> None:
        """
        Train policy network using the REINFORCE algorithm.

        Args:
            cartpole: cartpole environment
            num_episodes: number of episodes to run
            use_causality: whether to use the "causality trick" when defining loss function
            use_baseline: whether to use a baseline when defining the loss function
        """
        self.reset()
        cartpole.env.reset(seed=0)

        reward_history = []
        for episode in tqdm(range(num_episodes)):
            # Sample cartpole episode and get list of rewards and log probabilities
            # of the actions taken
            rewards, log_probs = self.sample_episode(cartpole)
            reward_history.append(sum(rewards))

            returns = []
            ##### YOUR CODE STARTS HERE #####
            # Compute `returns`, a list of discounted returns [R_0, R_1, ...]
            # where R_t = \sum_{t'=t}^{T-1} \gamma^(t'-t) r_t' 
            # Hint: use self.gamma
            raise NotImplementedError("Need to implement code here.")
            ###### YOUR CODE ENDS HERE ######
            assert len(returns) == len(rewards)
            returns = torch.tensor(returns)

            loss = []
            if use_causality:
                if use_baseline:
                    ##### YOUR CODE STARTS HERE #####
                    # Modify `returns` using a baseline that is the average of the previously
                    # computed returns, and scale by standard deviation for normalization.
                    raise NotImplementedError("Need to implement code here.")
                    ###### YOUR CODE ENDS HERE ######

                ##### YOUR CODE STARTS HERE #####
                # Compute the list `loss` of each time step loss where instead of using
                # the full episode return we use the tail return only
                raise NotImplementedError("Need to implement code here.")
                ###### YOUR CODE ENDS HERE ######
            else:
                ##### YOUR CODE STARTS HERE #####
                # Compute the list `loss` of each time step loss using the vanilla
                # approach where the full episode return is used.
                raise NotImplementedError("Need to implement code here.")
                ###### YOUR CODE ENDS HERE ######
            loss = torch.stack(loss).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Visualize reward history
        self.plot_rewards(reward_history)