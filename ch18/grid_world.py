import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.colors import LinearSegmentedColormap

W = LinearSegmentedColormap.from_list('w', ["w", "b"], N=256)

class GridWorld:
    def __init__(self, size=4, absorbing_states={(0, 0)}):
        """
        A gridworld environment with absorbing states at [0, 0] and [size - 1, size - 1].
        Args:
            size (int): the dimension of the grid in each direction, should be < 20 for render capability
            cell_reward (float): the reward return after extiting any non absorbing state
        """
        self.state_value = np.zeros((size, size))
        self.size = np.min((size, 20))
        self.actions = {
            0: [1, 0],   # north
            1: [-1, 0],  # south
            2: [0, -1],  # west
            3: [0, 1],   # east
        }
        if not absorbing_states:
            raise ValueError("Must provide absorbing states.")
        for state in absorbing_states:
            if (state[0] < 0 or state[0] >= size or state[1] < 0 or state[1] >= size):
                raise ValueError("Absorbing states must be within the grid bounds.")
        self.absorbing_states = absorbing_states
        return

    def reset(self):
        self.state_value = np.zeros((self.size, self.size))
        return

    def step(self, state, action):
        """
        Compute the next state and reward given the current state and action.
        The reward is 0 if the current state is an absorbing state and othewise
        it is -1.
        """
        if state in self.absorbing_states:
            # Absorbing state gets zero reward
            return state, 0

        next_state = (state[0] + action[0], state[1] + action[1])
        reward = -1
        # out of bounds north-south
        if next_state[0] < 0 or next_state[0] >= self.size:
            next_state = state
        # out of bounds east-west
        elif next_state[1] < 0 or next_state[1] >= self.size:
            next_state = state

        return next_state, reward

    def render(self, state_value=None, policy=None, title=None):
        """
        Displays the current value table of mini gridworld environment
        """
        values = state_value if state_value is not None else self.state_value
        if (values.shape != (self.size, self.size)):
            raise ValueError("state_value is wrong size")
        fig, ax = plt.subplots(figsize=(self.size, self.size))
        if title is not None:
            ax.set_title(title)
        sn.heatmap(values, annot=True, fmt=".1f", cmap=W,
                   linewidths=1, linecolor="black", cbar=False)
        if policy is not None:
            for i in range(self.size):
                for j in range(self.size):
                    for action_idx, action in self.actions.items():
                        scale = 0.25
                        p = scale * policy[i, j, action_idx]
                        base = np.array([i + 0.5, j + 0.5])
                        delta = np.array(action)
                        base += 0.25 * delta
                        plt.arrow(base[0], base[1], p * delta[0], p * delta[1], 
                                  color='r', width=0.1 * policy[i, j, action_idx], alpha=0.5)
        plt.show()
        return fig, ax