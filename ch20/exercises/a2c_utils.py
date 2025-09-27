import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
import os
import gymnasium as gym

class LunarLander(object):
    def __init__(self, video_file_directory="lunar_lander_outputs"):
        self.video_file_directory = video_file_directory
        os.makedirs(video_file_directory, exist_ok=True)

        # Create environment
        self.env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
        self.env.reset(seed=0)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

class MultivariateNormalDiag(eqx.Module):
    """
    Multivariate normal distribution with diagonal covariance.
    """
    mean: jnp.array # mean of the distribution
    std: jnp.array  # standard deviation

    @jax.jit
    def sample(self, key: jax.random.PRNGKey, shape=()):
        """
        Generate a random sample from the multivariate normal distribution.
        """
        return self.mean + self.std * jax.random.normal(key, shape + self.mean.shape)

    @jax.jit
    def log_prob(self, x: np.ndarray):
        """
        Compute the log of the probability density function evaluate at x.
        """
        return jnp.sum(jax.scipy.stats.norm.logpdf(x, self.mean, self.std), -1)

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