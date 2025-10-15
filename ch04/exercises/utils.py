from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely import box
from shapely.geometry import Polygon, LineString, Point

class Workspace2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic.
    """
    def __init__(self, width: float, height: float):
        self.lower = np.array([0, 0]) # state space lower bound
        self.upper = np.array([width, height]) # state space upper bound
        self.obstacles = [] # list of Polygons

    def set_obstacles(self, obstacles: list[Polygon]):
        self.obstacles = obstacles

    def generate_random_box_obstacles(self, num_obs, min_obs_size, max_obs_size):
        """
        Generate planning problem within the 2D workspace by defining
        a set of obstacles at random.
        """
        x_margin = self.width * 0.1
        y_margin = self.height * 0.1
        obs_corners_x = np.random.uniform(-x_margin, self.width + x_margin, num_obs)
        obs_corners_y = np.random.uniform(-y_margin, self.height + y_margin, num_obs)
        obs_lower_corners = np.vstack([obs_corners_x, obs_corners_y]).T
        obs_sizes = np.random.uniform(min_obs_size, max_obs_size, (num_obs, 2))
        obs_upper_corners = obs_lower_corners + obs_sizes
        obstacles = list(zip(obs_lower_corners, obs_upper_corners))
        self.obstacles = [box(obs[0][0], obs[0][1], obs[1][0], obs[1][1]) for obs in obstacles]

    @property
    def width(self):
        return self.upper[0] - self.lower[0]

    @property
    def height(self):
        return self.upper[1] - self.lower[1]

    def in_statespace(self, x: np.ndarray):
        """
        Checks if the point is inside the valid statespace
        """
        return all(x <= self.upper) and all(x >= self.lower)

    def random_sample(self):
        """
        Compute a random sample point within the statespace
        """
        return self.lower + np.random.rand(2)*(self.upper - self.lower)

    def is_free(self, x, margin=0.01):
        """
        Verifies that point is not inside any obstacles by some margin
        that is scaled by the max workspace dimension.
        """
        scale = np.max([self.width, self.height])
        for obs in self.obstacles:
            if Point(x).distance(obs) < scale * margin:
                return False
        return True

    def is_free_line(self, x1, x2):
        """
        Checks if line from `x1` to `x2` is collision free.
        """
        line = LineString([x1, x2])
        for obstacle in self.obstacles:
            if line.intersects(obstacle):
                return False
        return True

    def free_random_sample(self):
        """
        Compute a random sample point within the statespace
        """
        while True:
            x = self.random_sample()
            if (self.is_free(x)):
                return x

    def plot(self, fig_num=0, fig_size=6):
        """
        Plots the space and its obstacles.
        """
        fig = plt.figure(fig_num, figsize=[fig_size, fig_size])
        ax = fig.add_subplot(111, aspect='equal')
        for obs in self.obstacles:
            ax.add_patch(
            patches.Polygon(np.array(obs.exterior.coords)))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))

class Node:
    def __init__(self, point: np.ndarray, 
                 parent: Optional[int] = None, 
                 cost: Optional[float] = float('inf'), 
                 idx: Optional[int] = None):
        self.point = point
        self.parent = parent
        self.cost = cost
        self.idx = idx