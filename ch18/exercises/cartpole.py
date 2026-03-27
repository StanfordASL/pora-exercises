import gymnasium as gym
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math, torch, random, datetime
import cv2
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Literal, Tuple

def display_local_video_in_notebook(filename, size=(600, 400)):
    try:
        get_ipython()
        from IPython.display import HTML
    except (NameError, ImportError):
        raise RuntimeError("`display_local_video_in_notebook` must be called from jupyter/colab.")

    import base64
    import os
    # Re-encode video to x264.
    os.system(f"ffmpeg -y -i {filename} -vcodec libx264 {filename}.x264.mp4 -hide_banner -loglevel error")
    os.replace(filename + ".x264.mp4", filename)
    # Convert to base64 for display in notebook.
    with open(filename, "rb") as f:
        video_data = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()
    display(
        HTML(f"""
            <video width="{size[0]}" height="{size[1]}" controls autoplay loop>
                <source type="video/mp4" src="{video_data}">
                Your browser does not support the video tag.
            </video>
        """))

class Agent(ABC):
    def __init__(self, 
        state_dim: int, 
        action_dim: int, 
        name: Literal="agent", 
        use_gpu: bool=False,
        ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.gamma = 0.99 # discount factor
        self.learning_rate = 0.001
        self.batch_size = 128
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.name = name
    
    @abstractmethod
    def policy(self, state: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def plot_rewards(reward_history: List[int], ylabel: str='Total Reward') -> None:
        plt.figure()
        plt.plot(reward_history)
        plt.xlabel('Episode')
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.title('Learning Curve')
        plt.show()

class Cartpole():
    def __init__(self, file_directory: Literal="cartpole_outputs") -> None:
        self.file_directory = file_directory
        os.makedirs(file_directory, exist_ok=True)

        # Create environment
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.env.reset(seed=0)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 2

    def print_environment_info(self):
        print(f'Observation space: {self.env.observation_space}')
        print(f'Action space: {self.env.action_space}')

    def simulate(
        self, 
        agent: Agent, 
        num_episodes: int, 
        visualize: bool=False, 
        max_steps: int=None,
        ) -> None:
        """ 
        Rollout a particular policy in the cartpole environment.

        Args:
            agent: agent with a policy
            num_episodes: number of episodes to run
            visualize: whether to show visualization
            max_steps: max number of steps in episode
        """
        self.env.reset(seed=0)
        rewards = [] # for episode cumulative reward
        video_filename = f"{self.file_directory}/{agent.name}_cartpole_sim.mp4" \
                        if visualize else None
        for episode in tqdm(range(num_episodes)):
            episode_reward = 0
            obs, info = self.env.reset()
            terminated, truncated = False, False
            t = 0
            while not terminated and not truncated:
                # Render video at specified interval of episodes
                if episode == 0 and video_filename is not None:
                    # For episode 0 and t=0 start a video to store data
                    if t == 0:
                        video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*"mp4v"), 50, (600, 400))
                    video.write(self.env.render())

                # Evalute policy from the observation, apply action in the environment
                action = agent.policy(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action[0])
                episode_reward += reward

                truncated = True if max_steps is not None and t >= max_steps else truncated
                t += 1

            rewards.append(episode_reward)

        print("--- Reward Statistics ---")
        print(f'Average: {np.mean(rewards)}')
        print(f'Standard deviation: {np.std(rewards)}')
        print(f'Minimum: {np.min(rewards)}')
        print(f'Maximum: {np.max(rewards)}')

        if video_filename is not None:
            video.release()
            display_local_video_in_notebook(video_filename)

class Basic(Agent):
    """
    Basic agent whose policy is to simply move in the direction the pole is leaning.
    """
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__(state_dim, action_dim, name="basic")

    def policy(self, state: np.ndarray) -> np.ndarray:
        """
        Implement a simple controller that moves the cart in the direction the pole
        is leaning.
        """
        _, _, theta, _ = state # extract pole angle from environment
        left, right = np.array([0]), np.array([1])
        return left if theta < 0 else right
